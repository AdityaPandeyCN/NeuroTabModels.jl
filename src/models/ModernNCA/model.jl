"""
    ModernNCAModel

Lux container holding the feature embedding and the backbone encoder. The
embedding lives inside the model (rather than in front of it as for other
architectures) because reference rows must go through the same pipeline as
queries.
"""
struct ModernNCAModel{E,B,L<:LossType} <: LuxCore.AbstractLuxContainerLayer{(:embedding, :backbone)}
    embedding::E
    backbone::B
    cfg::ModernNCAConfig
    outsize::Int
    loss::L
end

"""
    _backbone(cfg, ins)

Linear, then `n_blocks` × (BN, Dense(relu), Dropout, Dense), then BN.
"""
function _backbone(cfg::ModernNCAConfig, ins::Int)
    layers = Any[Dense(ins => cfg.d_embedding)]
    for _ in 1:cfg.n_blocks
        push!(layers, BatchNorm(cfg.d_embedding))
        push!(layers, Dense(cfg.d_embedding => cfg.d_block, relu))
        cfg.dropout > 0 && push!(layers, Dropout(cfg.dropout))
        push!(layers, Dense(cfg.d_block => cfg.d_embedding))
    end
    cfg.n_blocks > 0 && push!(layers, BatchNorm(cfg.d_embedding))
    return Chain(layers...)
end

function _build_model(cfg, embedding, ins, outsize, loss)
    loss isa Union{MSE,MAE,LogLoss,MLogLoss} ||
        throw(ArgumentError("ModernNCA does not support $(nameof(typeof(loss)))"))
    return ModernNCAModel(embedding, _backbone(cfg, ins), cfg, Int(outsize), loss)
end

_temperature(m::ModernNCAModel) = max(m.cfg.temperature, m.cfg.eps)

# Chunk-major layout. A key set `(rows, N)` is stored as `(rows, chunk, N ÷ chunk)` plus
# a `(rows, N mod chunk)` tail, so a loop over chunks indexes with one scalar `k`. Plain
# Julia on CPU/GPU; inside a Reactant `@trace for` the scalar is traced and the index
# lowers to a dynamic slice, so the loop is a `stablehlo.while`, never unrolled.
_stack(x::AbstractMatrix, chunk::Int) = (k = fld(size(x, 2), chunk);
    (reshape(x[:, 1:(k * chunk)], size(x, 1), chunk, k), x[:, (k * chunk + 1):end]))
_stack(y::AbstractVector, chunk::Int) = (k = fld(length(y), chunk);
    (reshape(y[1:(k * chunk)], chunk, k), y[(k * chunk + 1):end]))
_block(x::AbstractArray{<:Any,3}, k) = x[:, :, k]
_block(y::AbstractMatrix, k) = y[:, k]

"""
    _pairwise_dist(q, k, ϵ) -> Matrix

`(num_keys, batch)` Euclidean distances between `q` `(d, batch)` and `k`
`(d, num_keys)`, via ‖q‖² + ‖k‖² − 2qᵀk so the only heavy operation is one GEMM.
"""
function _pairwise_dist(q::AbstractMatrix, k::AbstractMatrix, ϵ::Float32)
    q2 = sum(abs2, q; dims=1)
    k2 = sum(abs2, k; dims=1)
    return sqrt.(max.(0.0f0, k2' .+ q2 .- 2.0f0 .* (k' * q)) .+ ϵ)
end

_diag_inf(i, j, d) = ifelse(i == j, typemax(typeof(d)), d)
_mask_diag(d::AbstractMatrix) =
    _diag_inf.(reshape(1:size(d, 1), :, 1), reshape(1:size(d, 2), 1, :), d)

"""
    _scores(m, q, k; mask_self=false) -> (d, s)

Distances `d` and scores `s = -d / temperature`, both `(num_keys, batch)`.
`mask_self` sends the diagonal to `-Inf` so a query cannot attend to itself
(training, where the batch is part of the key set). `d` is returned unmasked
because the backward needs it.
"""
function _scores(m::ModernNCAModel, q, k; mask_self::Bool=false)
    d = _pairwise_dist(q, k, m.cfg.eps)
    return d, -(mask_self ? _mask_diag(d) : d) ./ _temperature(m)
end

"""
    _target_layout(loss, y)

Store targets per loss: a `(1, N)` row for regression/binary, class codes for
multiclass (a `(K, N)` one-hot is only built per chunk).
"""
_target_layout(::Union{MSE,MAE,LogLoss}, y) = reshape(y, 1, length(y))
_target_layout(::MLogLoss, y) = y

"""
    _targets(loss, y, outsize)

`(outsize, num_keys)` target block for a key block `y` in loss layout.
"""
_targets(::Union{MSE,MAE,LogLoss}, y, _) = y
_targets(::MLogLoss, y, n::Int) =
    ((k, c) -> ifelse(k == c, 1.0f0, 0.0f0)).(
        reshape(UInt32(1):UInt32(n), :, 1), reshape(y, 1, :))

_targets(m::ModernNCAModel, y) = _targets(m.loss, y, m.outsize)

"""
    _finalize(loss, p)

Map the weighted target average to what each loss expects: a mean (MSE), a
probability (LogLoss), or a class distribution (MLogLoss).
"""
_finalize(::Union{MSE,MAE}, p) = p
_finalize(::LogLoss, p) = (p = clamp.(p, 1.0f-6, 1.0f0 - 1.0f-6); log.(p ./ (1.0f0 .- p)))
_finalize(::MLogLoss, p) = log.(clamp.(p, 1.0f-7, Inf32))

"""
    _softmax_acc(zq, outsize) -> (running_max, denominator, numerator)

Accumulator for a softmax-weighted target average, each `(·, batch)`, folded
one key block at a time.
"""
function _softmax_acc(zq, outsize::Int)
    B = size(zq, 2)
    return (fill!(similar(zq, 1, B), -Inf32), fill!(similar(zq, 1, B), 0.0f0),
        fill!(similar(zq, outsize, B), 0.0f0))
end

"""
    _softmax_fold(acc, s, yk)

Fold score block `s` `(num_keys, batch)` with key targets `yk`
`(outsize, num_keys)`. Rescales the accumulator whenever the running max moves,
so the result equals a dense softmax while only this block is live.
"""
function _softmax_fold(acc, s, yk)
    running_max, denominator, numerator = acc
    block_max = maximum(s; dims=1)
    w = exp.(s .- block_max)
    merged = max.(running_max, block_max)
    old_scale, new_scale = exp.(running_max .- merged), exp.(block_max .- merged)
    return (merged, denominator .* old_scale .+ sum(w; dims=1) .* new_scale,
        numerator .* old_scale .+ (yk * w) .* new_scale)
end

"""
    _softmax_result(acc) -> (p, lse)

Weighted target average and per-query log-sum-exp.
"""
function _softmax_result(acc)
    running_max, denominator, numerator = acc
    return numerator ./ denominator, running_max .+ log.(denominator)
end

"""
    _encode(m, x, ps, st) -> (z, st)

Embedding then backbone. Returns `z` `(d_embedding, batch)` and the new state.
"""
function _encode(m::ModernNCAModel, x, ps, st)
    x, st_embedding = m.embedding(x, ps.embedding, st.embedding)
    z, st_backbone = m.backbone(x, ps.backbone, st.backbone)
    return z, (embedding=st_embedding, backbone=st_backbone)
end

"""
    _encode_all(m, (x3, xt), ps, st) -> (z3, zt)

Encode a chunk-major key set into `(d_embedding, chunk, nfull)` plus the tail.
"""
function _encode_all(m::ModernNCAModel, (x3, xt), ps, st)
    d = m.cfg.d_embedding
    z3 = similar(x3, d, size(x3, 2), size(x3, 3))
    @trace track_numbers = false for k in 1:size(x3, 3)
        z3[:, :, k] = first(_encode(m, x3[:, :, k], ps, st))
    end
    zt = size(xt, 2) == 0 ? similar(xt, d, 0) : first(_encode(m, xt, ps, st))
    return z3, zt
end

"""
    Corpus(x, y, info)

The full reference set used as keys outside training, chunk-major (see
[`_stack`](@ref)): features `x = (x3, xt)` on device, targets `y = (y3, yt)` in
loss layout, and a cached encoding `z`. `z` is (re)computed when
`info[:nrounds]` differs from `encoded_at`: during `fit`, parameters change
only in `fit_iter!`, which bumps `nrounds`, so the eval callback re-encodes once
per round; `infer_dataloader` encodes once up front. Inside a compiled graph
the encoding is recomputed per call and never cached. Marked as a Functors leaf
so device moves of `(x, corpus)` batches leave the resident corpus alone.
"""
mutable struct Corpus{X,Y,I}
    x::X
    y::Y
    info::I
    z::Any
    encoded_at::Int
end

Corpus(x, y, info) = Corpus(x, y, info, nothing, -1)

Functors.@leaf Corpus

function _keys(m::ModernNCAModel, corpus::Corpus, ps, st)
    round = corpus.info[:nrounds]
    corpus.encoded_at == round && return corpus.z
    z = _encode_all(m, corpus.x, ps, st)
    within_compile() && return z
    corpus.z, corpus.encoded_at = z, round
    return z
end

"""
    _attend_keys(m, zq, (z3, zt), (y3, yt)) -> acc

Fold the query encodings `zq` over an encoded key set, one chunk at a time.
"""
function _attend_keys(m::ModernNCAModel, zq, (z3, zt), (y3, yt))
    acc = _softmax_acc(zq, m.outsize)
    @trace track_numbers = false for k in 1:size(z3, 3)
        acc = _softmax_fold(acc, last(_scores(m, zq, z3[:, :, k])), _targets(m, _block(y3, k)))
    end
    size(zt, 2) == 0 && return acc
    return _softmax_fold(acc, last(_scores(m, zq, zt)), _targets(m, yt))
end

"""
    (m::ModernNCAModel)((x, corpus::Corpus), ps, st)

Eval / inference forward: attend from the encoded query batch over the full
corpus, one chunk at a time.
"""
function (m::ModernNCAModel)((x, corpus)::Tuple{Any,Corpus}, ps, st)
    zq, st = _encode(m, x, ps, st)
    p, _ = _softmax_result(_attend_keys(m, zq, _keys(m, corpus, ps, st), corpus.y))
    return _finalize(m.loss, p), st
end

"""
    (m::ModernNCAModel)((x, cand_x, cand_y, y), ps, st)

Training forward: the query batch attends over `[itself (self-masked); candidates]`.
`cand_x` is `(ins, chunk, nfull)` and `cand_y` `(chunk, nfull)` as produced by
[`ModernNCALoader`](@ref). Candidates are encoded and attended per chunk inside
[`_attend_train`](@ref), which has a custom backward.
"""
function (m::ModernNCAModel)((x, cand_x, cand_y, y)::Tuple{Any,Any,Any,Any}, ps, st)
    zq, st = _encode(m, x, ps, st)
    p, st, _ = _attend_train(m, zq, vec(y), cand_x, cand_y, ps, st)
    return _finalize(m.loss, p), st
end

_train_targets(m::ModernNCAModel, y) = _targets(m, _target_layout(m.loss, y))

"""
    _attend_train(m, zq, yq, cand_x, cand_y, ps, st; sts=nothing) -> (p, st, lse)

Online-softmax attention for training. Keys are `zq` itself (diagonal masked)
followed by `cand_x` encoded chunk by chunk with the current `ps` in train mode.
Also returns the per-query log-sum-exp. With `sts` a vector, the layer state
before each chunk is pushed onto it (the rrule needs it to recompute chunks
exactly). Under Reactant the chunk loop is traced with one checkpoint per
chunk, so Enzyme recomputes a chunk in the backward instead of keeping every
chunk's `(chunk, B)` scores: the same memory bound the rrule gives Zygote.
"""
function _attend_train(m::ModernNCAModel, zq, yq, cand_x, cand_y, ps, st; sts=nothing)
    acc = _softmax_acc(zq, m.outsize)
    acc = _softmax_fold(acc, _scores(m, zq, zq; mask_self=true)[2], _train_targets(m, yq))
    nfull = size(cand_x, 3)
    @trace track_numbers = false checkpointing = Periodic(max(nfull, 1)) for k in 1:nfull
        sts === nothing || push!(sts, st)
        zc, st = _encode(m, cand_x[:, :, k], ps, st)
        acc = _softmax_fold(acc, last(_scores(m, zq, zc)), _train_targets(m, cand_y[:, k]))
    end
    p, lse = _softmax_result(acc)
    return p, st, lse
end

"""
    _score_grads(m, q, k, d, dS) -> (dq, dk)

Gradients of `s = -d/T`, `d = sqrt(‖q-k‖² + ε)` given `dS` `(num_keys, batch)`,
using ∂d/∂q = (q-k)/d. The `max(0, ·)` clamp in `_pairwise_dist` is treated as
inactive.
"""
function _score_grads(m::ModernNCAModel, q, k, d, dS)
    G = dS ./ (d .* _temperature(m))
    return k * G .- q .* sum(G; dims=1), q * G' .- k .* sum(G; dims=2)'
end

"""
    ChainRulesCore.rrule(::typeof(_attend_train), ...)

FlashAttention-style backward: the forward kept only `p` and `lse`; each key
block is recomputed, `dS = P ∘ (Yᵀ dp − D)` with `D = rowsum(p ∘ dp)`, and
candidate chunks are pulled back through the encoder one at a time.
"""
function ChainRulesCore.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(_attend_train),
    m::ModernNCAModel, zq, yq, cand_x, cand_y, ps, st)
    sts = Any[]
    p, st_out, lse = _attend_train(m, zq, yq, cand_x, cand_y, ps, st; sts)
    function attend_train_pullback(Δ)
        dp = unthunk(Δ[1])
        D = sum(p .* dp; dims=1)

        d, s = _scores(m, zq, zq; mask_self=true)
        dS = exp.(s .- lse) .* (_train_targets(m, yq)' * dp .- D)
        dzq = .+(_score_grads(m, zq, zq, d, dS)...)

        dps = ZeroTangent()
        for k in 1:size(cand_x, 3)
            cx = cand_x[:, :, k]
            zc, enc_pb = rrule_via_ad(cfg, p_ -> first(_encode(m, cx, p_, sts[k])), ps)
            d, s = _scores(m, zq, zc)
            dS = exp.(s .- lse) .* (_train_targets(m, cand_y[:, k])' * dp .- D)
            dq, dk = _score_grads(m, zq, zc, d, dS)
            dzq = dzq .+ dq
            dps = dps + enc_pb(dk)[2]
        end
        return NoTangent(), NoTangent(), dzq, NoTangent(), NoTangent(), NoTangent(), dps, NoTangent()
    end
    return (p, st_out, lse), attend_train_pullback
end
