using Lux
using Random
using NNlib

"""
    LinearEmbeddings(n_features, d_embedding)

Embeds each continuous feature via a learned affine transformation: `w_j * x_j + b_j`.
Produces a `(d_embedding, n_features, batch)` tensor.

# Arguments
- `n_features::Int`: Number of input features.
- `d_embedding::Int`: Embedding dimension per feature.
"""
struct LinearEmbeddings <: Lux.AbstractLuxLayer
    n_features::Int
    d_embedding::Int
end

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

"""
    LinearReLUEmbeddings(n_features, d_embedding=32)

`LinearEmbeddings` followed by element-wise ReLU activation.

# Arguments
- `n_features::Int`: Number of input features.
- `d_embedding::Int`: Embedding dimension per feature (default `32`).
"""
struct LinearReLUEmbeddings <: Lux.AbstractLuxContainerLayer{(:layer,)}
    layer::LinearEmbeddings
end

function LinearReLUEmbeddings(n_features::Int, d_embedding::Int=32)
    return LinearReLUEmbeddings(LinearEmbeddings(n_features, d_embedding))
end

function (l::LinearReLUEmbeddings)(x, ps, st)
    y, st_l = l.layer(x, ps.layer, st.layer)
    return NNlib.relu.(y), (layer=st_l,)
end