_is_training(st) = st.training === Val(true) || st.training === true

"""
    CarryMask(layer)

Pass a padding mask through a layer that only acts on the feature matrix:
`(x, mask) ↦ (layer(x), mask)`. Unmasked `x` is forwarded unchanged.
"""
struct CarryMask{L} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer::L
end
(l::CarryMask)(x::AbstractArray, ps, st) = l.layer(x, ps, st)
function (l::CarryMask)((x, mask)::Tuple, ps, st)
    y, st_ = l.layer(x, ps, st)
    return (y, mask), st_
end

"""
    MaskSkip(layer)

Skip-add that threads a padding mask: `(x, mask) ↦ (x + f(x), mask)`.
"""
struct MaskSkip{L} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer::L
end
function (l::MaskSkip)(x::AbstractMatrix, ps, st)
    y, st_ = l.layer(x, ps, st)
    return x .+ y, st_
end
function (l::MaskSkip)((x, mask)::Tuple, ps, st)
    y, st_ = l.layer((x, mask), ps, st)
    h = y isa Tuple ? y[1] : y
    return (x .+ h, mask), st_
end

"""
    MaskedBatchNorm(chs, act=identity; epsilon=1f-5, momentum=0.1f0)

BatchNorm over the last dimension of `(channels, tokens)`.

When the input is `(x, valid)` with `valid` a per-token flag, mean/variance and
running stats use only valid tokens so zero-padded group-buffer slots do not leak.
Unmasked `x` is Lux's `BatchNorm`.

Dense / Dropout in the same chain should be wrapped in [`CarryMask`](@ref);
residual skips should use [`MaskSkip`](@ref).
"""
struct MaskedBatchNorm{B<:BatchNorm} <: LuxCore.AbstractLuxWrapperLayer{:bn}
    bn::B
end
MaskedBatchNorm(chs::Int, act=identity; kwargs...) = MaskedBatchNorm(BatchNorm(chs, act; kwargs...))

function _token_w(valid, x::AbstractMatrix)
    return reshape(eltype(x).(valid), 1, size(x, 2))
end

function _moments(x::AbstractMatrix, w::AbstractMatrix, n)
    μ = sum(x .* w; dims=2) ./ n
    σ² = sum(((x .- μ) .^ 2) .* w; dims=2) ./ n
    return μ, σ²
end

function _bn_apply(bn::BatchNorm, x::AbstractMatrix, valid, ps, st)
    T = eltype(x)
    ϵ = T(bn.epsilon)
    γ = reshape(ps.scale, :, 1)
    β = reshape(ps.bias, :, 1)
    if _is_training(st)
        w = _token_w(valid, x)
        nvalid = sum(w)
        n = max(nvalid, one(T))
        μ, σ² = _moments(x, w, n)
        y = bn.activation.(γ .* (x .- μ) ./ sqrt.(σ² .+ ϵ) .+ β)
        # An all-pad batch leaves running stats alone; arithmetic (no `if`) so Reactant can trace it.
        # Momentum and the n / (n - 1) variance correction follow LuxLib's batchnorm.
        m = T(bn.momentum) * ifelse(nvalid > zero(T), one(T), zero(T))
        rm = (1 - m) .* st.running_mean .+ m .* vec(μ)
        rv = (1 - m) .* st.running_var .+ (m * n / max(n - 1, one(T))) .* vec(σ²)
        return y, (; running_mean=rm, running_var=rv, training=st.training)
    end
    μ = reshape(st.running_mean, :, 1)
    σ² = reshape(st.running_var, :, 1)
    y = bn.activation.(γ .* (x .- μ) ./ sqrt.(σ² .+ ϵ) .+ β)
    return y, st
end

(l::MaskedBatchNorm)(x::AbstractMatrix, ps, st) = l.bn(x, ps, st)
function (l::MaskedBatchNorm)((x, valid)::Tuple, ps, st)
    y, st_ = _bn_apply(l.bn, x, valid, ps, st)
    return (y, valid), st_
end
