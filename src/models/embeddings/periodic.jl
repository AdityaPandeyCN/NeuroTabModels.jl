using Lux
using Random
using NNlib

struct Periodic <: Lux.AbstractLuxLayer
    n_features::Int
    n_frequencies::Int
    sigma::Float32
end

# NOTE: Struct defines Periodic(n, k, sigma) automatically.

function Lux.initialparameters(rng::AbstractRNG, l::Periodic)
    bound = l.sigma * 3f0
    w = clamp.(l.sigma .* randn(rng, Float32, l.n_frequencies, l.n_features), -bound, bound)
    return (weight=w,)
end

Lux.initialstates(::AbstractRNG, ::Periodic) = (;)

function (l::Periodic)(x::AbstractMatrix, ps, st)
    x_r = reshape(x, 1, size(x, 1), size(x, 2))
    w = reshape(ps.weight, size(ps.weight, 1), size(ps.weight, 2), 1)
    z = 2f0 * Float32(π) .* w .* x_r
    return vcat(cos.(z), sin.(z)), st
end

struct PeriodicEmbeddings{P,L,A} <: Lux.AbstractLuxLayer
    periodic::P
    linear::L
    activation::A
    lite::Bool
end

function PeriodicEmbeddings(
    n_features::Int,
    d_embedding::Int=24;
    n_frequencies::Int=48,
    frequency_init_scale::Float32=0.01f0,
    activation::Bool=true,
    lite::Bool=false,
)
    if lite && !activation
        error("lite=true is allowed only when activation=true")
    end

    periodic = Periodic(n_features, n_frequencies, frequency_init_scale)

    if lite
        linear = Dense(2 * n_frequencies => d_embedding)
    else
        linear = NLinear(n_features, 2 * n_frequencies, d_embedding)
    end

    act = activation ? NNlib.relu : identity
    return PeriodicEmbeddings(periodic, linear, act, lite)
end

function Lux.initialparameters(rng::AbstractRNG, m::PeriodicEmbeddings)
    return (periodic=Lux.initialparameters(rng, m.periodic),
            linear=Lux.initialparameters(rng, m.linear))
end

function Lux.initialstates(rng::AbstractRNG, m::PeriodicEmbeddings)
    return (periodic=Lux.initialstates(rng, m.periodic),
            linear=Lux.initialstates(rng, m.linear))
end

function (m::PeriodicEmbeddings)(x::AbstractMatrix, ps, st)
    h, st_p = m.periodic(x, ps.periodic, st.periodic)

    h, st_l = if m.lite
        d_in, n, b = size(h)
        h_flat = reshape(h, d_in, n * b)
        out_flat, st_sub = m.linear(h_flat, ps.linear, st.linear)
        reshape(out_flat, size(out_flat, 1), n, b), st_sub
    else
        m.linear(h, ps.linear, st.linear)
    end

    return m.activation.(h), (periodic=st_p, linear=st_l)
end