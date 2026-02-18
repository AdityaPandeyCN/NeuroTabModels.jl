using Lux
using Random
using NNlib

struct LinearEmbeddings <: Lux.AbstractLuxLayer
    n_features::Int
    d_embedding::Int
end

# NOTE: Struct automatically provides LinearEmbeddings(n::Int, d::Int)
# No need to redefine it.

function Lux.initialparameters(rng::AbstractRNG, l::LinearEmbeddings)
    limit = Float32(l.d_embedding)^(-0.5f0)
    weight = (rand(rng, Float32, l.d_embedding, l.n_features) .* 2f0 .* limit) .- limit
    bias = (rand(rng, Float32, l.d_embedding, l.n_features) .* 2f0 .* limit) .- limit
    return (weight=weight, bias=bias)
end

Lux.initialstates(::AbstractRNG, ::LinearEmbeddings) = (;)

function (l::LinearEmbeddings)(x::AbstractMatrix, ps, st)
    x_r = reshape(x, 1, size(x, 1), size(x, 2))
    w = reshape(ps.weight, size(ps.weight, 1), size(ps.weight, 2), 1)
    b = reshape(ps.bias, size(ps.bias, 1), size(ps.bias, 2), 1)
    return (w .* x_r) .+ b, st
end

struct LinearReLUEmbeddings <: Lux.AbstractLuxLayer
    layer::LinearEmbeddings
end

# Convenience constructor for users (different signature than struct)
function LinearReLUEmbeddings(n_features::Int, d_embedding::Int=32)
    return LinearReLUEmbeddings(LinearEmbeddings(n_features, d_embedding))
end

Lux.initialparameters(rng::AbstractRNG, l::LinearReLUEmbeddings) = Lux.initialparameters(rng, l.layer)
Lux.initialstates(rng::AbstractRNG, l::LinearReLUEmbeddings) = Lux.initialstates(rng, l.layer)

function (l::LinearReLUEmbeddings)(x, ps, st)
    y, st = l.layer(x, ps, st)
    return NNlib.relu.(y), st
end