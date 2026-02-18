using Lux: AbstractLuxLayer, WrappedFunction
using LuxCore
using LuxLib: batched_matmul
using Random: AbstractRNG
using NNlib: relu

# ─── Init helpers (CPU only) ─────────────────────────────────────────

function _init_rsqrt_uniform(rng::AbstractRNG, dims, d::Int)
    s = Float32(1 / sqrt(d))
    return s .* (2f0 .* rand(rng, Float32, dims...) .- 1f0)
end

function _init_scaling(rng::AbstractRNG, dims, init::Symbol)
    if init == :ones
        return ones(Float32, dims...)
    elseif init == :normal
        return randn(rng, Float32, dims...)
    elseif init == :random_signs
        # Float32 cast to avoid Float64 intermediates on GPU
        return Float32.(2 .* (rand(rng, Float32, dims...) .> 0.5f0) .- 1)
    else
        error("Unknown scaling init: $init")
    end
end

# ─── Named Activation ────────────────────────────────────────────────
# Essential for Reactant: ensures XLA can compile the graph without 
# closure capture issues.
_broadcast_relu(x) = relu.(x)

# ─── EnsembleView: (D, B) → (D, K, B) ────────────────────────────────
# Dispatch separates trace paths, avoiding runtime control flow.

struct EnsembleView <: AbstractLuxLayer
    k::Int
end

LuxCore.initialparameters(::AbstractRNG, ::EnsembleView) = (;)
LuxCore.initialstates(::AbstractRNG, ::EnsembleView) = (;)

# 2D Input: Expand
function (m::EnsembleView)(x::AbstractMatrix, ps, st)
    D, B = size(x)
    return repeat(reshape(x, D, 1, B), 1, m.k, 1), st
end

# 3D Input: Pass through
function (m::EnsembleView)(x::AbstractArray{T,3}, ps, st) where {T}
    return x, st
end

# ─── LinearEfficientEnsemble (BatchEnsemble) ─────────────────────────
# Uses LuxLib.batched_matmul and constant propagation for branching.

struct LinearEfficientEnsemble <: AbstractLuxLayer
    in_features::Int
    out_features::Int
    k::Int
    use_r::Bool
    use_s::Bool
    use_bias::Bool
    ensemble_bias::Bool
    scaling_init::Symbol
end

function LinearEfficientEnsemble(in_f::Int, out_f::Int;
        k::Int, scaling_init::Symbol=:random_signs,
        ensemble_scaling_in::Bool=true, ensemble_scaling_out::Bool=true,
        bias::Bool=true, ensemble_bias::Bool=true)
    ensemble_bias && @assert bias "ensemble_bias requires bias=true"
    return LinearEfficientEnsemble(
        in_f, out_f, k,
        ensemble_scaling_in, ensemble_scaling_out,
        bias, ensemble_bias, scaling_init
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, m::LinearEfficientEnsemble)
    d = (; weight = _init_rsqrt_uniform(rng, (m.out_features, m.in_features), m.in_features))
    
    if m.use_r
        d = merge(d, (; r = _init_scaling(rng, (m.in_features, m.k), m.scaling_init)))
    end
    
    if m.use_s
        d = merge(d, (; s = _init_scaling(rng, (m.out_features, m.k), m.scaling_init)))
    end
    
    if m.use_bias
        b0 = _init_rsqrt_uniform(rng, (m.out_features,), m.in_features)
        # Broadcast init logic
        bias_val = m.ensemble_bias ? repeat(b0, 1, m.k) : b0
        d = merge(d, (; bias = bias_val))
    end
    
    return d
end

LuxCore.initialstates(::AbstractRNG, ::LinearEfficientEnsemble) = (;)

function (m::LinearEfficientEnsemble)(x::AbstractArray{T,3}, ps, st) where {T}
    in_f, k, batch = size(x)

    # 1. R Scaling (Input Diversity)
    if m.use_r
        x = x .* reshape(ps.r, m.in_features, m.k, 1)
    end

    # 2. Shared Weights (Efficient MatMul)
    # Collapse K and Batch dimensions for a single large matmul
    x_flat = reshape(x, in_f, k * batch)
    out_flat = ps.weight * x_flat
    x = reshape(out_flat, m.out_features, k, batch)

    # 3. S Scaling (Output Diversity)
    if m.use_s
        x = x .* reshape(ps.s, m.out_features, m.k, 1)
    end

    # 4. Bias
    if m.use_bias
        bias_k = m.ensemble_bias ? m.k : 1
        x = x .+ reshape(ps.bias, m.out_features, bias_k, 1)
    end

    return x, st
end

# ─── LinearEnsemble (Independent Heads) ──────────────────────────────
# Uses batched_matmul for high-performance independent linear layers.

struct LinearEnsemble <: AbstractLuxLayer
    in_features::Int
    out_features::Int
    k::Int
    use_bias::Bool
end

LinearEnsemble(in_f::Int, out_f::Int, k::Int; bias::Bool=true) =
    LinearEnsemble(in_f, out_f, k, bias)

function LuxCore.initialparameters(rng::AbstractRNG, m::LinearEnsemble)
    d = (; weight = _init_rsqrt_uniform(rng, (m.out_features, m.in_features, m.k), m.in_features))
    if m.use_bias
        d = merge(d, (; bias = _init_rsqrt_uniform(rng, (m.out_features, m.k), m.in_features)))
    end
    return d
end

LuxCore.initialstates(::AbstractRNG, ::LinearEnsemble) = (;)

function (m::LinearEnsemble)(x::AbstractArray{T,3}, ps, st) where {T}
    # Input: (In, K, Batch)
    # Weights: (Out, In, K)
    
    # Permute for batched_matmul: (In, Batch, K)
    xp = permutedims(x, (1, 3, 2))
    
    # (Out, In, K) * (In, Batch, K) -> (Out, Batch, K)
    out = batched_matmul(ps.weight, xp)
    
    # Permute back: (Out, K, Batch)
    out = permutedims(out, (1, 3, 2))
    
    if m.use_bias
        out = out .+ reshape(ps.bias, m.out_features, m.k, 1)
    end
    return out, st
end

# ─── ScaleEnsemble (TabM-mini) ───────────────────────────────────────

struct ScaleEnsemble <: AbstractLuxLayer
    k::Int; d::Int; init::Symbol
end

ScaleEnsemble(k::Int, d::Int; init::Symbol=:random_signs) = ScaleEnsemble(k, d, init)

LuxCore.initialparameters(rng::AbstractRNG, m::ScaleEnsemble) =
    (; weight = _init_scaling(rng, (m.d, m.k), m.init))
LuxCore.initialstates(::AbstractRNG, ::ScaleEnsemble) = (;)

function (m::ScaleEnsemble)(x::AbstractArray{T,3}, ps, st) where {T}
    return x .* reshape(ps.weight, m.d, m.k, 1), st
end

# ─── MeanEnsemble ────────────────────────────────────────────────────
# Averaging over ensemble dimension K. 
# Uses sum/k instead of mean for better compiler support.

struct MeanEnsemble <: AbstractLuxLayer end

LuxCore.initialparameters(::AbstractRNG, ::MeanEnsemble) = (;)
LuxCore.initialstates(::AbstractRNG, ::MeanEnsemble) = (;)

function (::MeanEnsemble)(x::AbstractArray{T,3}, ps, st) where {T}
    k = size(x, 2)
    # Sum over dim 2 (K), then divide
    return dropdims(sum(x; dims=2); dims=2) ./ T(k), st
end