module TabM

export TabMConfig

using Random
using Lux
using LuxCore
using NNlib

import ..Losses: get_loss_type, GaussianMLE
import ..Models: Architecture
import ..Embeddings: PeriodicEmbeddings, LinearEmbeddings, PiecewiseLinearEmbeddings

include("layers.jl")

# ─── Reactant-safe flatten for embeddings ────────────────────────
# (d_emb, nfeats, batch) → (d_emb * nfeats, batch)
_flatten_emb(x::AbstractArray{T,3}) where {T} = reshape(x, :, size(x, 3))
# Pass-through for 2D (in case embedding already flattens)
_flatten_emb(x::AbstractMatrix) = x

# ─── Backbone builders ──────────────────────────────────────────

function _batch_ensemble_backbone(; d_in, n_blocks, d_block, dropout, k, scaling_init)
    layers = []
    for i in 1:n_blocks
        ind = i == 1 ? d_in : d_block
        si = i == 1 ? scaling_init : :ones
        push!(layers, LinearEfficientEnsemble(ind, d_block;
            k=k, scaling_init=si,
            ensemble_scaling_in=true, ensemble_scaling_out=true,
            bias=true, ensemble_bias=true))
        push!(layers, WrappedFunction(_broadcast_relu))
        dropout > 0 && push!(layers, Dropout(dropout))
    end
    return layers
end

function _mini_ensemble_backbone(; d_in, n_blocks, d_block, dropout, k, scaling_init)
    layers = Any[ScaleEnsemble(k, d_in; init=scaling_init)]
    for i in 1:n_blocks
        ind = i == 1 ? d_in : d_block
        push!(layers, LinearEfficientEnsemble(ind, d_block;
            k=k, ensemble_scaling_in=false, ensemble_scaling_out=false,
            bias=true, ensemble_bias=false))
        push!(layers, WrappedFunction(_broadcast_relu))
        dropout > 0 && push!(layers, Dropout(dropout))
    end
    return layers
end

function _packed_ensemble_backbone(; d_in, n_blocks, d_block, dropout, k)
    layers = []
    for i in 1:n_blocks
        ind = i == 1 ? d_in : d_block
        push!(layers, LinearEnsemble(ind, d_block, k))
        push!(layers, WrappedFunction(_broadcast_relu))
        dropout > 0 && push!(layers, Dropout(dropout))
    end
    return layers
end

# ─── TabMConfig ─────────────────────────────────────────────────

struct TabMConfig <: Architecture
    k::Int
    n_blocks::Int
    d_block::Int
    dropout::Float64
    arch_type::Symbol
    scaling_init::Symbol
    MLE_tree_split::Bool
    use_embeddings::Bool
    d_embedding::Int
    embedding_type::Symbol
end

function TabMConfig(; kwargs...)
    args = Dict{Symbol,Any}(
        :k => 32,
        :n_blocks => 3,
        :d_block => 512,
        :dropout => 0.1,
        :arch_type => :tabm,
        :scaling_init => :random_signs,
        :MLE_tree_split => false,
        :use_embeddings => false,
        :d_embedding => 24,
        :embedding_type => :periodic,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @warn "Following $(length(args_ignored)) provided arguments will be ignored: $(join(args_ignored, ", "))."

    args_default = setdiff(keys(args), keys(kwargs))
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments set to default: $(join(args_default, ", "))."

    for arg in intersect(keys(args), keys(kwargs))
        args[arg] = kwargs[arg]
    end

    return TabMConfig(
        args[:k],
        args[:n_blocks],
        args[:d_block],
        args[:dropout],
        Symbol(args[:arch_type]),
        Symbol(args[:scaling_init]),
        args[:MLE_tree_split],
        args[:use_embeddings],
        args[:d_embedding],
        Symbol(args[:embedding_type]),
    )
end

function (config::TabMConfig)(; nfeats, outsize)
    k = config.k
    d_block = config.d_block

    # ── 1. Feature Processing ────────────────────────────────────
    if config.use_embeddings
        if config.embedding_type == :periodic
            emb_layer = PeriodicEmbeddings(nfeats, config.d_embedding)
            d_out_per_feature = config.d_embedding
        elseif config.embedding_type == :linear
            emb_layer = LinearEmbeddings(nfeats, config.d_embedding)
            d_out_per_feature = config.d_embedding
        elseif config.embedding_type == :piecewise
            emb_layer = PiecewiseLinearEmbeddings(nfeats, config.d_embedding)
            d_out_per_feature = config.d_embedding
        else
            error("Unsupported embedding type: $(config.embedding_type). Use :periodic, :linear, or :piecewise")
        end

        # Flatten (d_emb, nfeats, B) → (d_emb * nfeats, B)
        feature_processor = Chain(emb_layer, WrappedFunction(_flatten_emb))
        d_in = nfeats * d_out_per_feature

        # Paper Section A.2: with embeddings, init first R from N(0,1)
        effective_scaling_init = :normal
    else
        feature_processor = BatchNorm(nfeats)
        d_in = nfeats
        effective_scaling_init = config.scaling_init
    end

    # ── 2. Backbone ──────────────────────────────────────────────
    bb = if config.arch_type == :tabm
        _batch_ensemble_backbone(; d_in, n_blocks=config.n_blocks,
            d_block, dropout=config.dropout, k,
            scaling_init=effective_scaling_init)
    elseif config.arch_type == :tabm_mini
        _mini_ensemble_backbone(; d_in, n_blocks=config.n_blocks,
            d_block, dropout=config.dropout, k,
            scaling_init=effective_scaling_init)
    elseif config.arch_type == :tabm_packed
        _packed_ensemble_backbone(; d_in, n_blocks=config.n_blocks,
            d_block, dropout=config.dropout, k)
    else
        error("Unknown arch_type: $(config.arch_type)")
    end

    # ── 3. Head ──────────────────────────────────────────────────
    head = if config.MLE_tree_split && outsize == 2
        split_out = outsize ÷ 2
        Parallel(
            vcat,
            Chain(LinearEnsemble(d_block, split_out, k), MeanEnsemble()),
            Chain(LinearEnsemble(d_block, split_out, k), MeanEnsemble()),
        )
    else
        Chain(LinearEnsemble(d_block, outsize, k), MeanEnsemble())
    end

    # ── 4. Full Model ────────────────────────────────────────────
    return Chain(
        feature_processor,
        EnsembleView(k),
        bb...,
        head,
    )
end

end # module