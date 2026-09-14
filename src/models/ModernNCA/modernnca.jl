module ModernNCA

export ModernNCAConfig

import ..Models
import ..Models: Architecture, NeuroTabModel
import ..Losses: LossType, MSE, MAE, LogLoss, MLogLoss

using Lux
using Lux: Functors, MLDataDevices
using ReactantCore: @trace, within_compile, Periodic
using LuxCore
using NNlib: relu
using ChainRulesCore: ChainRulesCore, RuleConfig, HasReverseMode, NoTangent, ZeroTangent,
    rrule_via_ad, unthunk
using Random: randperm, AbstractRNG
using StatsBase: sample
using DataFrames: AbstractDataFrame, select
using CategoricalArrays

"""
    ModernNCAConfig(; d_embedding=128, n_blocks=2, d_block=256,
                     dropout=0.1, temperature=1.0, sample_rate=0.8,
                     corpus_chunk_size=2048, eps=1f-8)

Hyperparameters for ModernNCA. Pass as the `arch` argument to `NeuroTabRegressor`
or `NeuroTabClassifier`.

# Arguments
- `d_embedding`: encoder output dimension.
- `n_blocks`, `d_block`: number and hidden width of MLP blocks after the linear layer.
- `dropout`: dropout rate inside each block; skipped when `<= 0`.
- `temperature`: softmax temperature on negative pairwise distances.
- `sample_rate`: fraction of the batch complement sampled as candidates per
  training step (Stochastic Neighborhood Sampling). `1.0` uses the full complement.
- `corpus_chunk_size`: keys encoded and attended per chunk, in training and
  inference. This is the memory knob.
- `eps`: numerical floor under `sqrt` and as a `temperature` lower bound.
"""
struct ModernNCAConfig <: Architecture
    d_embedding::Int
    n_blocks::Int
    d_block::Int
    dropout::Float32
    temperature::Float32
    sample_rate::Float32
    corpus_chunk_size::Int
    eps::Float32
end

function ModernNCAConfig(;
    d_embedding::Int=128, n_blocks::Int=2, d_block::Int=256,
    dropout::Real=0.1, temperature::Real=1.0, sample_rate::Real=0.8,
    corpus_chunk_size::Int=2048, eps::Real=1.0f-8,
)
    corpus_chunk_size > 0 || throw(ArgumentError("corpus_chunk_size must be positive"))
    return ModernNCAConfig(d_embedding, n_blocks, d_block, Float32(dropout),
        Float32(temperature), Float32(sample_rate), corpus_chunk_size, Float32(eps))
end

include("model.jl")

"""
    (cfg::ModernNCAConfig)(; ins, outsize, loss=MSE)

Build a `ModernNCAModel` with a `NoOpLayer` embedding. Prefer
`Models.build_chain` so the real feature embedding lives inside the model.

# Arguments
- `ins`: number of input features after embedding (or raw features if `NoOpLayer`).
- `outsize`: output size (`1` for MSE/MAE/LogLoss, `K` for MLogLoss).
- `loss`: `MSE`, `MAE`, `LogLoss`, or `MLogLoss`.
"""
(cfg::ModernNCAConfig)(; ins, outsize, loss::LossType=MSE(), kwargs...) =
    _build_model(cfg, NoOpLayer(), ins, outsize, loss)

"""
    Models.build_chain(cfg::ModernNCAConfig, embedding; ins, outsize, loss)

Build `ModernNCAModel` with `embedding` inside the Lux container so query and
corpus rows share one encoder.

# Arguments
- `cfg`: ModernNCA config.
- `embedding`: feature-embedding layer.
- `ins`: embedding output dimension (encoder input size).
- `outsize`: output size (`1` for MSE/MAE/LogLoss, `K` for MLogLoss).
- `loss`: `MSE`, `MAE`, `LogLoss`, or `MLogLoss`.
"""
Models.build_chain(cfg::ModernNCAConfig, embedding; ins, outsize, loss, kwargs...) =
    _build_model(cfg, embedding, ins, outsize, loss)

"""
    ModernNCALoader

Training iterator yielding `((x, cand_x, cand_y, y), y)`. Query rows follow a
shuffled epoch permutation; `n_cand` candidates are resampled every step from
the complement of the query batch and delivered chunk-major: `cand_x`
`(ins, chunk, n_cand ÷ chunk)`, `cand_y` `(chunk, n_cand ÷ chunk)`.

# Arguments
- `full_x`: feature matrix `(ins, N)`.
- `full_y`: encoded targets of length `N`.
- `batchsize`: query rows per step.
- `n_cand`: candidate rows sampled from the batch complement; a multiple of `chunk`.
- `chunk`: `corpus_chunk_size`.
- `rng`: sampler for the epoch permutation and candidate indices.
- `dev`: device.
- `host`: if `true`, `full_x`/`full_y` stay on the host, rows are gathered there
  and each batch is moved with `dev`; otherwise the corpus is device-resident
  and only the index vectors are moved.
"""
struct ModernNCALoader{X,Y,R<:AbstractRNG,D}
    full_x::X
    full_y::Y
    batchsize::Int
    n_cand::Int
    chunk::Int
    rng::R
    dev::D
    host::Bool
end

# Reactant compiles every eager `getindex` on a device array, so gather on the host there.
_host_gather(::Any) = false
_host_gather(::MLDataDevices.ReactantDevice) = true

function _gather(l::ModernNCALoader, idx)
    l.host && return l.dev(l.full_x[:, idx]), l.dev(l.full_y[idx])
    idx = l.dev(idx)
    return l.full_x[:, idx], l.full_y[idx]
end

Base.length(l::ModernNCALoader) = fld(size(l.full_x, 2), l.batchsize)

"""
    Base.iterate(l::ModernNCALoader, state=nothing)

One training step. `state = (perm, start)`. Candidates are drawn from `perm`
outside the window `start:stop` with an index skip, so the complement is never
materialised.

# Arguments
- `l`: loader.
- `state`: `(perm, start)` after the first call; `nothing` starts a new epoch.
"""
function Base.iterate(l::ModernNCALoader, state=nothing)
    n = size(l.full_x, 2)
    perm, start = state === nothing ? (randperm(l.rng, n), 1) : state
    stop = start + l.batchsize - 1
    stop > n && return nothing

    x, y = _gather(l, perm[start:stop])

    if l.n_cand > 0
        js = sample(l.rng, 1:(n - l.batchsize), l.n_cand; replace=false)
        cand_x, cand_y = _gather(l, perm[@. ifelse(js < start, js, js + l.batchsize)])
    else
        cand_x, cand_y = l.dev(similar(l.full_x, size(l.full_x, 1), 0)), l.dev(similar(l.full_y, 0))
    end
    k = l.n_cand ÷ l.chunk
    cand_x, cand_y = reshape(cand_x, size(cand_x, 1), l.chunk, k), reshape(cand_y, l.chunk, k)
    return ((x, cand_x, cand_y, y), y), (perm, stop + 1)
end

"""
    build_corpus(df, feature_names, target_name, loss, scalers) -> (full_x, full_y)

Feature matrix `(ins, N)` as `Float32` and encoded targets.

# Arguments
- `df`: training data frame.
- `feature_names`: columns used as features.
- `target_name`: target column.
- `loss`: `MSE`, `MAE`, `LogLoss`, or `MLogLoss`.
- `scalers`: target mean/std for MSE/MAE, or `nothing`.
"""
function build_corpus(df::AbstractDataFrame, feature_names, target_name,
    loss::LossType, scalers)
    full_x = permutedims(Matrix{Float32}(select(df, collect(feature_names))))
    return full_x, _encode_targets(df, target_name, loss, scalers)
end

"""
    _encode_targets(df, target_name, loss, scalers)

MLogLoss: 1-based `UInt32` codes. LogLoss: `Float32` in `{0, 1}`. MSE/MAE:
`Float32`, scaled.
"""
_encode_targets(df, target_name, ::MLogLoss, _) =
    UInt32.(CategoricalArrays.levelcode.(df[!, target_name]))

function _encode_targets(df, target_name, ::LogLoss, _)
    col = df[!, target_name]
    eltype(col) <: CategoricalValue || return Float32.(col)
    length(CategoricalArrays.levels(col)) == 2 ||
        error("For `loss=:logloss`, target must have exactly 2 classes.")
    return Float32.(CategoricalArrays.levelcode.(col) .- 1)
end

function _encode_targets(df, target_name, ::Union{MSE,MAE}, scalers)
    y = Float32.(df[!, target_name])
    isnothing(scalers) || (y .= (y .- scalers.mu) ./ scalers.sigma)
    return y
end

"""
    Models.train_dataloader(cfg::ModernNCAConfig, ...)

Build `ModernNCALoader` and stash the raw corpus on `m.info[:nca_ref]`.
`n_cand = floor(sample_rate × (N − batchsize))`, rounded down to a multiple of
`corpus_chunk_size`. `sample_rate ≥ 1` uses the full complement; `batchsize == N`
gives `n_cand = 0`. Backends: `:zygote` (memory-bounded backward via the
`ChainRulesCore` rrule) and `:reactant` (the same chunk loop traced with one
checkpoint per chunk). Weights, offsets, and group padding are rejected.

# Arguments
- `cfg`: ModernNCA config.
- `m`: fitted model wrapper.
- `df`: training data frame.
- `feature_names`
- `target_name`
- `loss`
- `scalers`
- `batchsize`
- `dev`: device
- `rng`
"""
function Models.train_dataloader(cfg::ModernNCAConfig, m::NeuroTabModel, ::Any, df;
    feature_names, target_name, loss, scalers, batchsize, dev, rng,
    weight_name=nothing, offset_name=nothing, group_name=nothing, backend=:zygote, kwargs...)
    backend in (:zygote, :reactant) ||
        throw(ArgumentError("ModernNCA training supports :zygote or :reactant (got $backend)"))
    for (val, name) in
        ((weight_name, :weight_name), (offset_name, :offset_name), (group_name, :group_name))
        isnothing(val) || throw(ArgumentError("ModernNCA does not support `$name`"))
    end
    cx, cy = build_corpus(df, feature_names, target_name, loss, scalers)
    m.info[:nca_ref] = (cx=cx, cy=cy)
    n = size(cx, 2)
    batchsize = min(batchsize, n)
    pool = n - batchsize
    n_cand = pool == 0 ? 0 :
             cfg.sample_rate >= 1.0f0 ? pool :
             max(Int(floor(cfg.sample_rate * pool)), 1)
    n_cand = n_cand ÷ cfg.corpus_chunk_size * cfg.corpus_chunk_size
    host = _host_gather(dev)
    return ModernNCALoader(host ? cx : dev(cx), host ? cy : dev(cy), batchsize, n_cand,
        cfg.corpus_chunk_size, rng, dev, host)
end

function _corpus(m::ModernNCAModel, info, dev)
    ref, chunk = info[:nca_ref], m.cfg.corpus_chunk_size
    return Corpus(dev(_stack(ref.cx, chunk)), dev(_stack(_target_layout(m.loss, ref.cy), chunk)), info)
end

"""
    Models.eval_dataloader(m::ModernNCAModel, info, data, dev, ps, st)

Attach a [`Corpus`](@ref) to every eval batch `(x, y, ...)`; the encoding
refreshes per round.

# Arguments
- `m`: ModernNCA model.
- `info`: fit metadata; must contain `:nca_ref` and `:nrounds`.
- `data`: default eval iterator of `(x, y, ...)`.
- `dev`: device.
- `ps`, `st`: unused; encoding happens on the first forward of the round.
"""
function Models.eval_dataloader(m::ModernNCAModel, info, data, dev, ::Any, ::Any)
    corpus = _corpus(m, info, dev)
    return Iterators.map(d -> ((d[1], corpus), Base.tail(d)...), data)
end

"""
    Models.infer_dataloader(m::ModernNCAModel, info, data, dev, ps, st)

Attach a [`Corpus`](@ref) to every inference batch, encoded once here with
`ps`, `st`. With `grouped=true`, preserve each batch's row mask.

# Arguments
- `m`: ModernNCA model.
- `info`: fit metadata; must contain `:nca_ref` and `:nrounds`.
- `data`: default inference iterator.
- `dev`: device.
- `ps`, `st`: parameters and (test-mode) states used to encode the corpus.
- `backend`: AD backend; selects how the encoder is run (see `Models.compile_fn`).
- `grouped`: if `true`, keep each batch's row mask.
"""
function Models.infer_dataloader(m::ModernNCAModel, info, data, dev, ps, st;
    backend=:zygote, grouped::Bool=false)
    corpus = _corpus(m, info, dev)
    encode = Models.compile_fn(Val(backend), _encode_all, m, corpus.x, ps, st)
    corpus.z, corpus.encoded_at = encode(m, corpus.x, ps, st), info[:nrounds]
    return grouped ?
           Iterators.map(d -> ((d[1], corpus), d[2]), data) :
           Iterators.map(x -> (x, corpus), data)
end

end
