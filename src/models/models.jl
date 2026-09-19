module Models

export NeuroTabModel, Architecture, uses_batch_mask, MaskedModel
export MaskedBatchNorm, CarryMask, MaskSkip, GroupedDense, AttnResidual, ResidualScale
export Embeddings, EmbeddingLayer
export LinearEmbeddings, PeriodicEmbeddings, PiecewiseLinearEmbeddings
export BatchNormEmbeddings, LayerNormEmbeddings, TemporalEmbeddings, IdentityEmbedding
export AbstractNumericalEmbedding, AbstractTemporalEmbedding, AbstractEmbedding
export NeuroTreeConfig, MLPConfig, MLPAttnConfig, NeuroTreeAttnConfig
export ResNetConfig, TabMConfig, MOETreeConfig, ModernNCAConfig

using ..Losses
using Lux: Chain
using LuxCore: AbstractLuxContainerLayer
using NNlib

"""
    Architecture

Abstract type for architecture **configs** (`MLPConfig`, `ResNetConfig`, …).

A config holds hyperparameters. Calling `(config)(; ins, outsize)` instantiates
the Lux architecture: a `Chain` or other `AbstractLuxLayer`. Learners
(`NeuroTabRegressor`, `NeuroTabClassifier`) take the config, not the
instantiated layer.
"""
abstract type Architecture end

const activation_dict = Dict{Symbol,Function}(
    :relu => NNlib.relu, :gelu => NNlib.gelu, :sigmoid => NNlib.sigmoid_fast, :tanh => NNlib.tanh_fast
)

function get_activation(act::Symbol)
    haskey(activation_dict, act) ||
        error("Unknown activation: $act. Supported: $(sort(collect(keys(activation_dict))))")
    return activation_dict[act]
end

"""
    uses_batch_mask(layer) -> Bool

Whether `layer` consumes a `(x, mask)` tuple so padded group-buffer slots can be ignored
in batch-level attention. Default is `false`; architectures that mix across observations
override this.
"""
uses_batch_mask(::Any) = false
uses_batch_mask(c::Chain) = any(uses_batch_mask, c.layers)

"""
    MaskedModel(embed, core)

Assemble embeddings and a core backbone that may take an optional padding mask.

Embeddings still see only `x`. When the input is `(x, w)`, this layer runs
`embed(x)` then `core((z, w))`. Used instead of `Chain(embed, core)` when the
core mixes across observations and needs to ignore padded group-buffer slots.

Embeddings are per-observation maps, so a pad column cannot contaminate a
real column there; the mask is only required where the core reduces or mixes
across the batch (attention, `MaskedBatchNorm`). See the "Padding and masks"
design page.
"""
struct MaskedModel{E,C} <: AbstractLuxContainerLayer{(:embed, :core)}
    embed::E
    core::C
end

uses_batch_mask(::MaskedModel) = true

function (m::MaskedModel)(x::AbstractArray, ps, st)
    z, st_e = m.embed(x, ps.embed, st.embed)
    y, st_c = m.core(z, ps.core, st.core)
    return y, (; embed=st_e, core=st_c)
end
function (m::MaskedModel)((x, w)::Tuple, ps, st)
    z, st_e = m.embed(x, ps.embed, st.embed)
    y, st_c = m.core((z, w), ps.core, st.core)
    return y, (; embed=st_e, core=st_c)
end

import ..Losses: masked_input
masked_input(::MaskedModel, x, w) = (x, w)

"""
    train_dataloader(arch, m, default, df; kwargs...)

Per-architecture hook: return the dataloader `fit` should use. Default returns
the tabular `default` unchanged; retrieval-style archs override to substitute
their own iterator (e.g. with a candidate corpus attached).

Overrides may write into `m.info` (e.g. to stash a corpus for inference).
"""
train_dataloader(::Architecture, ::Any, data, ::Any; kwargs...) = data

"""
    build_chain(arch, embed_chain; ins, outsize)

Assemble the Lux chain that `fit` will train: `MaskedModel(embed_chain, core)`
when the core consumes a batch mask, `Chain(embed_chain, core)` otherwise.
Architectures can override this when the embedding must be part of the model
itself.
"""
function build_chain(arch::Architecture, embed_chain; ins, outsize, kwargs...)
    core = arch(; ins, outsize, kwargs...)
    return uses_batch_mask(core) ? MaskedModel(embed_chain, core) : Chain(embed_chain, core)
end

"""
    infer_dataloader(chain, info, data, dev, ps, st)

Per-architecture hook: return the per-batch iterator `infer` should use.
Default returns `data` unchanged; retrieval-style archs override to wrap each
batch with extra inputs (e.g. a candidate corpus). Dispatched on
`typeof(m.chain)` so arch modules can override on their concrete model type.
"""
infer_dataloader(::Any, ::Any, data, ::Any, ::Any, ::Any; kwargs...) = data

"""
    compile_fn(::Val{backend}, f, args...) -> callable

`f` compiled for `args` on backends that compile ahead of time (Reactant);
`f` itself elsewhere.
"""
compile_fn(::Val, f, args...) = f

"""
    stopgrad(x)

`x` with its gradient treated as zero. Identity on plain arrays; the Reactant
extension lowers it to `Ops.ignore_derivatives` so Enzyme never differentiates
through `x`.
"""
stopgrad(x) = x

# Loops over data, by kind. Chunked arrays keep their chunks along the last dimension.
# Each kind is a plain loop here; a backend can take one over by adding a method for its
# own array type (the Reactant extension lowers them to a single traced loop). The step
# `f` is a plain function and everything it needs beyond the chunks is passed as `args`:
# a backend that traces the loop turns `f`'s arguments into loop arguments, and a closure
# over model data cannot be lowered that way.

"""
    chunkat(x, k)

Chunk `k` of `x`: the slice at index `k` of the last dimension.
"""
chunkat(x::AbstractArray{<:Any,3}, k) = x[:, :, k]
chunkat(x::AbstractMatrix, k) = x[:, k]
chunksat(xs::Tuple, k) = map(x -> chunkat(x, k), xs)

nchunks(x::AbstractArray) = size(x, ndims(x))

"""
    mapchunks!(f, out, xs::Tuple, args...) -> out

Independent chunks: `out[:, :, k] = f(chunk k of each xs..., args...)`. Nothing is
carried from one chunk to the next.
"""
function mapchunks!(f, out::AbstractArray{<:Any,3}, xs::Tuple, args...)
    for k in 1:nchunks(out)
        out[:, :, k] = f(chunksat(xs, k)..., args...)
    end
    return out
end

"""
    foldchunks(f, state, xs::Tuple, args...; checkpoint=false) -> state

Sequential chunks: `state = f(state, chunk k of each xs..., args...)`. `state` must
keep its structure, array sizes and element types from one chunk to the next.
`checkpoint=true` asks a backend that differentiates the loop itself to keep one
checkpoint per chunk and recompute the rest in the backward.
"""
function foldchunks(f, state, xs::Tuple, args...; checkpoint::Bool=false)
    for k in 1:nchunks(first(xs))
        state = f(state, chunksat(xs, k)..., args...)
    end
    return state
end

"""
    istraced(x) -> Bool

Whether `x` is a placeholder a backend is tracing the model with, not real data.
"""
istraced(x) = false

"""
    eval_dataloader(chain, info, data, dev, ps, st)

Per-architecture hook: return the per-batch iterator the eval `CallBack` should
use. Default returns `data` unchanged; retrieval-style archs override to wrap
each batch's `x` with extra inputs (e.g. a candidate corpus) so that the eval
forward pass matches the training/inference signature. Dispatched on
`typeof(m.chain)`.
"""
eval_dataloader(::Any, ::Any, data, ::Any, ::Any, ::Any) = data

"""
    NeuroTabModel

The object containing the model and associated metadata.

# Fields

- `loss`: the loss functor used in training (`MSE()`, `LogLoss()`, `MLogLoss()`, `GaussianMLE()`).
- `chain`: the underlying `Lux.Chain` neural network.
- `info`: a `Dict{Symbol,Any}` of metadata such as `:feature_names`, `:target_levels`,
  and `:device`, plus the fitted parameters (`ps`) and state (`st`).
"""
struct NeuroTabModel{L<:LossType,C}
    loss::L
    chain::C
    info::Dict{Symbol,Any}
end

include("layers/layers.jl")
using .Layers: MaskedBatchNorm, CarryMask, MaskSkip, GroupedDense, AttnResidual, ResidualScale

include("embeddings/embeddings.jl")
using .Embeddings

include("MLP/mlp.jl")
using .MLP

include("ResNet/resnet.jl")
using .ResNet

include("NeuroTree/neurotrees.jl")
using .NeuroTrees

include("TabM/TabM.jl")
using .TabM

include("ModernNCA/modernnca.jl")
using .ModernNCA

end