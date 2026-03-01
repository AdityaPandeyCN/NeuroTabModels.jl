using Lux
using Random
using NNlib

_broadcast_relu_ple(x) = NNlib.relu.(x)

function _ple_activate(x::AbstractArray{T,3}, single_bin_mask) where {T}
    M, N, B = size(x)
    if M == 1
        return x
    end

    first_row = min.(view(x, 1:1, :, :), one(T))
    mid_rows = clamp.(view(x, 2:M-1, :, :), zero(T), one(T))
    last_row_raw = view(x, M:M, :, :)

    last_row = if isnothing(single_bin_mask)
        max.(last_row_raw, zero(T))
    else
        ifelse.(single_bin_mask, last_row_raw, max.(last_row_raw, zero(T)))
    end

    return cat(first_row, mid_rows, last_row; dims=1)
end

# ─── PiecewiseLinearEncoding (leaf layer, non-trainable) ─────────

struct PiecewiseLinearEncoding <: Lux.AbstractLuxLayer
    bins::Vector{Vector{Float32}}
    n_features::Int
    max_n_bins::Int
end

function PiecewiseLinearEncoding(bins::Vector{<:AbstractVector})
    @assert length(bins) > 0
    n_features = length(bins)
    n_bins_list = [length(b) - 1 for b in bins]
    max_n_bins = maximum(n_bins_list)
    bins_f32 = [Float32.(b) for b in bins]
    return PiecewiseLinearEncoding(bins_f32, n_features, max_n_bins)
end

Lux.initialparameters(::AbstractRNG, ::PiecewiseLinearEncoding) = (;)

function Lux.initialstates(::AbstractRNG, l::PiecewiseLinearEncoding)
    weight = zeros(Float32, l.max_n_bins, l.n_features)
    bias = zeros(Float32, l.max_n_bins, l.n_features)
    n_bins_list = [length(b) - 1 for b in l.bins]

    for (i, bin_edges) in enumerate(l.bins)
        bin_width = diff(bin_edges)
        w = 1f0 ./ bin_width
        b = -bin_edges[1:end-1] ./ bin_width
        nb = n_bins_list[i]
        weight[end, i] = w[end]
        bias[end, i] = b[end]
        if nb > 1
            weight[1:nb-1, i] = w[1:end-1]
            bias[1:nb-1, i] = b[1:end-1]
        end
    end

    sbm_vec = [nb == 1 for nb in n_bins_list]
    single_bin_mask = any(sbm_vec) ? reshape(sbm_vec, 1, :, 1) : nothing

    return (weight=weight, bias=bias, single_bin_mask=single_bin_mask)
end

function (l::PiecewiseLinearEncoding)(x::AbstractMatrix, ps, st)
    x_r = reshape(x, 1, size(x, 1), size(x, 2))
    w = reshape(st.weight, size(st.weight, 1), size(st.weight, 2), 1)
    b = reshape(st.bias, size(st.bias, 1), size(st.bias, 2), 1)
    h = b .+ w .* x_r
    return _ple_activate(h, st.single_bin_mask), st
end

# ─── PiecewiseLinearEmbeddings (container) ────────────────────────
# Fields named in the type parameter are auto-handled by Lux.
# linear0 can be `nothing` for version :A — Lux skips nothing fields.

struct PiecewiseLinearEmbeddings{L0,I,L} <: Lux.AbstractLuxContainerLayer{(:linear0, :encoding, :linear)}
    linear0::L0
    encoding::I
    linear::L
    use_activation::Bool
    version::Symbol
end

function PiecewiseLinearEmbeddings(
    bins::Vector{<:AbstractVector},
    d_embedding::Int;
    activation::Bool=false,
    version::Symbol=:B,
)
    @assert version in (:A, :B)
    n_features = length(bins)
    max_n_bins = maximum(length(b) - 1 for b in bins)

    encoding = PiecewiseLinearEncoding(bins)
    linear0 = (version == :B) ? LinearEmbeddings(n_features, d_embedding) : nothing
    linear = NLinear(n_features, max_n_bins, d_embedding; bias=(version == :A))

    return PiecewiseLinearEmbeddings(linear0, encoding, linear, activation, version)
end

# Lux auto-generates initialparameters/initialstates from container fields.
# But we need to override initialparameters to zero-init linear.weight for version B.
function Lux.initialparameters(rng::AbstractRNG, m::PiecewiseLinearEmbeddings)
    # Let Lux collect sublayer params automatically
    ps = Lux.initialparameters(rng, Lux.StatefulLuxLayer{true}(m, nothing, nothing))
    # Actually, for a container layer we can just call sublayer inits directly:
    ps_l0 = m.linear0 === nothing ? nothing : Lux.initialparameters(rng, m.linear0)
    ps_enc = Lux.initialparameters(rng, m.encoding)
    ps_lin = Lux.initialparameters(rng, m.linear)

    if m.version == :B
        ps_lin = merge(ps_lin, (weight=zeros(Float32, size(ps_lin.weight)...),))
    end

    return (linear0=ps_l0, encoding=ps_enc, linear=ps_lin)
end

function (m::PiecewiseLinearEmbeddings)(x::AbstractMatrix, ps, st)
    val_linear0 = nothing
    st_l0 = st.linear0

    if m.linear0 !== nothing
        val_linear0, st_l0 = m.linear0(x, ps.linear0, st.linear0)
    end

    h_enc, st_enc = m.encoding(x, ps.encoding, st.encoding)
    h_proj, st_lin = m.linear(h_enc, ps.linear, st.linear)

    h_final = val_linear0 === nothing ? h_proj : (val_linear0 .+ h_proj)

    if m.use_activation
        h_final = _broadcast_relu_ple(h_final)
    end

    return h_final, (linear0=st_l0, encoding=st_enc, linear=st_lin)
end