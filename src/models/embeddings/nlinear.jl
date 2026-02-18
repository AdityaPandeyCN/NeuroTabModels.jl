using Lux
using Random
using NNlib

struct NLinear <: Lux.AbstractLuxLayer
    n::Int
    in_features::Int
    out_features::Int
    use_bias::Bool
end

# Outer constructor with keyword argument (safe because signature differs from struct default)
function NLinear(n::Int, in_features::Int, out_features::Int; bias::Bool=true)
    return NLinear(n, in_features, out_features, bias)
end

function Lux.initialparameters(rng::AbstractRNG, l::NLinear)
    limit = Float32(l.in_features)^(-0.5f0)
    weight = (rand(rng, Float32, l.out_features, l.in_features, l.n) .* 2f0 .* limit) .- limit
    
    bias = nothing
    if l.use_bias
        bias = (rand(rng, Float32, l.out_features, 1, l.n) .* 2f0 .* limit) .- limit
    end
    
    return l.use_bias ? (weight=weight, bias=bias) : (weight=weight,)
end

Lux.initialstates(::AbstractRNG, ::NLinear) = (;)

function (l::NLinear)(x::AbstractArray{T, 3}, ps, st) where T
    x_perm = permutedims(x, (1, 3, 2))
    out = batched_mul(ps.weight, x_perm)
    
    if l.use_bias
        out = out .+ ps.bias
    end
    
    y = permutedims(out, (1, 3, 2))
    return y, st
end