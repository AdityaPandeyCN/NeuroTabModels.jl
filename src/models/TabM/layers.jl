#= TabM Ensemble Layers (Lux.jl)
Port of tabm.py v0.0.3 — https://github.com/yandex-research/tabm
Tensor convention: Python (B, K, D) → Julia (D, K, B)
=#

using Lux: AbstractLuxLayer, WrappedFunction
using LuxCore
using LuxLib: batched_matmul
using Random: AbstractRNG
using NNlib: relu

# ─── Init helpers ────────────────────────────────────────────────

# Uniform[-1/√d, 1/√d]
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
        return Float32.(2 .* (rand(rng, Float32, dims...) .> 0.5f0) .- 1)
    else
        error("Unknown scaling init: $init")
    end
end

# Chunk-based init: within each chunk, all values share one random sample per member.
# Used for first adapter when features have different representation sizes.
function _init_scaling_with_chunks(rng::AbstractRNG, dims::Tuple{Int,Int},
                                    init::Symbol, chunks::Vector{Int})
    d, k = dims
    @assert d == sum(chunks) "Chunks must sum to $d, got $(sum(chunks))"
    weight = zeros(Float32, d, k)
    row = 1
    for chunk_size in chunks
        val = _init_scaling(rng, (1, k), init)
        weight[row:row+chunk_size-1, :] .= repeat(val, chunk_size, 1)
        row += chunk_size
    end
    return weight
end

# ─── Reactant-safe activation ────────────────────────────────────

_broadcast_relu(x) = relu.(x)

# ─── EnsembleView: (D, B) → (D, K, B) ───────────────────────────

struct EnsembleView <: AbstractLuxLayer
    k::Int
end

LuxCore.initialparameters(::AbstractRNG, ::EnsembleView) = (;)
LuxCore.initialstates(::AbstractRNG, ::EnsembleView) = (;)

function (m::EnsembleView)(x::AbstractMatrix, ps, st)
    D, B = size(x)
    return repeat(reshape(x, D, 1, B), 1, m.k, 1), st
end

function (m::EnsembleView)(x::AbstractArray{T,3}, ps, st) where {T}
    @assert size(x, 2) == m.k "Expected K=$(m.k), got $(size(x, 2))"
    return x, st
end

# ─── LinearBatchEnsemble ─────────────────────────────────────────
# y = S ⊙ (W(R ⊙ x)) + bias
# W shared, R/S/bias per-member.
# scaling_init can be Symbol or Tuple{Symbol,Symbol} for (R, S).

struct LinearBatchEnsemble <: AbstractLuxLayer
    in_features::Int
    out_features::Int
    k::Int
    use_bias::Bool
    r_init::Symbol
    s_init::Symbol
    r_init_chunks::Union{Nothing, Vector{Int}}
end

function LinearBatchEnsemble(in_f::Int, out_f::Int;
        k::Int,
        scaling_init::Union{Symbol, Tuple{Symbol,Symbol}} = :random_signs,
        first_scaling_init_chunks::Union{Nothing, Vector{Int}} = nothing,
        bias::Bool = true)
    r_init, s_init = scaling_init isa Tuple ? scaling_init : (scaling_init, scaling_init)
    return LinearBatchEnsemble(in_f, out_f, k, bias, r_init, s_init, first_scaling_init_chunks)
end

function LuxCore.initialparameters(rng::AbstractRNG, m::LinearBatchEnsemble)
    weight = _init_rsqrt_uniform(rng, (m.out_features, m.in_features), m.in_features)

    r = if m.r_init_chunks !== nothing
        _init_scaling_with_chunks(rng, (m.in_features, m.k), m.r_init, m.r_init_chunks)
    else
        _init_scaling(rng, (m.in_features, m.k), m.r_init)
    end

    s = _init_scaling(rng, (m.out_features, m.k), m.s_init)

    d = (; weight, r, s)
    if m.use_bias
        # All k biases share same init (shared bias + k zero non-shared biases)
        b0 = _init_rsqrt_uniform(rng, (m.out_features,), m.in_features)
        d = merge(d, (; bias = repeat(reshape(b0, :, 1), 1, m.k)))
    end
    return d
end

LuxCore.initialstates(::AbstractRNG, ::LinearBatchEnsemble) = (;)

function (m::LinearBatchEnsemble)(x::AbstractArray{T,3}, ps, st) where {T}
    in_f, k, batch = size(x)
    x = x .* reshape(ps.r, m.in_features, m.k, 1)
    x = reshape(ps.weight * reshape(x, in_f, k * batch), m.out_features, k, batch)
    x = x .* reshape(ps.s, m.out_features, m.k, 1)
    if m.use_bias
        x = x .+ reshape(ps.bias, m.out_features, m.k, 1)
    end
    return x, st
end

# ─── SharedDense (for TabM-mini backbone) ────────────────────────
# Standard shared linear that broadcasts over K dimension.
# NOT BatchEnsemble — truly shared bias (out,), not per-member (out, k).

struct SharedDense <: AbstractLuxLayer
    in_features::Int
    out_features::Int
end

function LuxCore.initialparameters(rng::AbstractRNG, m::SharedDense)
    return (;
        weight = _init_rsqrt_uniform(rng, (m.out_features, m.in_features), m.in_features),
        bias = _init_rsqrt_uniform(rng, (m.out_features,), m.in_features),
    )
end

LuxCore.initialstates(::AbstractRNG, ::SharedDense) = (;)

function (m::SharedDense)(x::AbstractArray{T,3}, ps, st) where {T}
    d_in, k, batch = size(x)
    out = ps.weight * reshape(x, d_in, k * batch) .+ ps.bias
    return reshape(out, m.out_features, k, batch), st
end

# ─── LinearEnsemble (k independent heads) ────────────────────────

struct LinearEnsemble <: AbstractLuxLayer
    in_features::Int
    out_features::Int
    k::Int
    use_bias::Bool
end

LinearEnsemble(in_f::Int, out_f::Int, k::Int; bias::Bool = true) =
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
    xp = permutedims(x, (1, 3, 2))          # (in, k, batch) → (in, batch, k)
    out = batched_matmul(ps.weight, xp)      # (out, in, k) × (in, batch, k) → (out, batch, k)
    out = permutedims(out, (1, 3, 2))        # → (out, k, batch)
    if m.use_bias
        out = out .+ reshape(ps.bias, m.out_features, m.k, 1)
    end
    return out, st
end

# ─── ScaleEnsemble (ElementwiseAffine for TabM-mini) ─────────────
# Multiplicative (+ optional additive) per-member transform.
# TabM-mini uses bias=false.

struct ScaleEnsemble <: AbstractLuxLayer
    k::Int
    d::Int
    init::Symbol
    init_chunks::Union{Nothing, Vector{Int}}
    use_bias::Bool
end

function ScaleEnsemble(k::Int, d::Int;
        init::Symbol = :random_signs,
        init_chunks::Union{Nothing, Vector{Int}} = nothing,
        bias::Bool = false)
    return ScaleEnsemble(k, d, init, init_chunks, bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, m::ScaleEnsemble)
    weight = if m.init_chunks !== nothing
        _init_scaling_with_chunks(rng, (m.d, m.k), m.init, m.init_chunks)
    else
        _init_scaling(rng, (m.d, m.k), m.init)
    end
    d = (; weight)
    if m.use_bias
        d = merge(d, (; bias = zeros(Float32, m.d, m.k)))
    end
    return d
end

LuxCore.initialstates(::AbstractRNG, ::ScaleEnsemble) = (;)

function (m::ScaleEnsemble)(x::AbstractArray{T,3}, ps, st) where {T}
    w = reshape(ps.weight, m.d, m.k, 1)
    if m.use_bias
        return reshape(ps.bias, m.d, m.k, 1) .+ w .* x, st
    else
        return x .* w, st
    end
end

# ─── MeanEnsemble ────────────────────────────────────────────────
# sum/k instead of mean for Reactant compatibility.

struct MeanEnsemble <: AbstractLuxLayer end

LuxCore.initialparameters(::AbstractRNG, ::MeanEnsemble) = (;)
LuxCore.initialstates(::AbstractRNG, ::MeanEnsemble) = (;)

function (::MeanEnsemble)(x::AbstractArray{T,3}, ps, st) where {T}
    k = size(x, 2)
    return dropdims(sum(x; dims=2); dims=2) ./ T(k), st
end