module Losses

export get_loss_fn, get_loss_type
export LossType, MSE, MAE, LogLoss, MLogLoss, GaussianMLE, Tweedie
export reduce_pred

import Statistics: mean
import NNlib: logsigmoid, logsoftmax

abstract type LossType end
abstract type MSE <: LossType end
abstract type MAE <: LossType end
abstract type LogLoss <: LossType end
abstract type MLogLoss <: LossType end
abstract type GaussianMLE <: LossType end
abstract type Tweedie <: LossType end

"""
    reduce_pred(y)

Identity for 2D predictions. For 3D ensemble output `(D, K, B)`, averages over K → `(D, B)`.
"""
reduce_pred(y::AbstractMatrix) = y
reduce_pred(y::AbstractArray{T,3}) where {T} =
    dropdims(sum(y; dims=2); dims=2) ./ T(size(y, 2))

_repeat_K(a::AbstractMatrix, K::Int) =
    reshape(repeat(reshape(a, size(a, 1), 1, size(a, 2)), 1, K, 1), size(a, 1), K * size(a, 2))
_repeat_K(a::AbstractVector, K::Int) =
    vec(repeat(reshape(a, 1, length(a)), K, 1))

_handle_ensemble(pred::AbstractMatrix, arrays...) = pred, arrays
function _handle_ensemble(pred::AbstractArray{T,3}, arrays...) where {T}
    d, K, B = size(pred)
    return reshape(pred, d, K * B), map(a -> _repeat_K(a, K), arrays)
end

"""
    mse_loss(model, ps, st, data)

Mean squared error. `data` is a tuple of `(X, y)`, `(X, y, w)`, or `(X, y, w, offset)`.
"""
function mse_loss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_,) = _handle_ensemble(pred, data[2])
    p = vec(pred); y = vec(y_)
    return mean((p .- y) .^ 2), st_, NamedTuple()
end
function mse_loss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_) = _handle_ensemble(pred, data[2], data[3])
    p = vec(pred); y = vec(y_); w = vec(w_)
    return sum((p .- y) .^ 2 .* w) / sum(w), st_, NamedTuple()
end
function mse_loss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_, o_) = _handle_ensemble(pred, data[2], data[3], data[4])
    p = vec(pred) .+ vec(o_); y = vec(y_); w = vec(w_)
    return sum((p .- y) .^ 2 .* w) / sum(w), st_, NamedTuple()
end

"""
    mae_loss(model, ps, st, data)

Mean absolute error. `data` is a tuple of `(X, y)`, `(X, y, w)`, or `(X, y, w, offset)`.
"""
function mae_loss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_,) = _handle_ensemble(pred, data[2])
    p = vec(pred); y = vec(y_)
    return mean(abs.(p .- y)), st_, NamedTuple()
end
function mae_loss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_) = _handle_ensemble(pred, data[2], data[3])
    p = vec(pred); y = vec(y_); w = vec(w_)
    return sum(abs.(p .- y) .* w) / sum(w), st_, NamedTuple()
end
function mae_loss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_, o_) = _handle_ensemble(pred, data[2], data[3], data[4])
    p = vec(pred) .+ vec(o_); y = vec(y_); w = vec(w_)
    return sum(abs.(p .- y) .* w) / sum(w), st_, NamedTuple()
end

"""
    logloss(model, ps, st, data)

Binary cross-entropy (logit-space). `data` is a tuple of `(X, y)`, `(X, y, w)`, or `(X, y, w, offset)`.
"""
function logloss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_,) = _handle_ensemble(pred, data[2])
    p = vec(pred); y = vec(y_)
    return mean((1 .- y) .* p .- logsigmoid.(p)), st_, NamedTuple()
end
function logloss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_) = _handle_ensemble(pred, data[2], data[3])
    p = vec(pred); y = vec(y_); w = vec(w_)
    return sum(w .* ((1 .- y) .* p .- logsigmoid.(p))) / sum(w), st_, NamedTuple()
end
function logloss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_, o_) = _handle_ensemble(pred, data[2], data[3], data[4])
    p = vec(pred) .+ vec(o_); y = vec(y_); w = vec(w_)
    return sum(w .* ((1 .- y) .* p .- logsigmoid.(p))) / sum(w), st_, NamedTuple()
end

"""
    mlogloss(model, ps, st, data)

Multiclass cross-entropy (logit-space). `data` is a tuple of `(X, y)`, `(X, y, w)`, or `(X, y, w, offset)`.
`y` contains integer class labels.
"""
function mlogloss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_,) = _handle_ensemble(pred, data[2])
    k = size(pred, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(y_, 1, :)
    lsm = logsoftmax(pred; dims=1)
    return mean(-sum(y_oh .* lsm; dims=1)), st_, NamedTuple()
end
function mlogloss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_) = _handle_ensemble(pred, data[2], data[3])
    k = size(pred, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(y_, 1, :)
    lsm = logsoftmax(pred; dims=1)
    return sum(vec(-sum(y_oh .* lsm; dims=1)) .* vec(w_)) / sum(w_), st_, NamedTuple()
end
function mlogloss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_, o_) = _handle_ensemble(pred, data[2], data[3], data[4])
    pred = pred .+ o_
    k = size(pred, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(y_, 1, :)
    lsm = logsoftmax(pred; dims=1)
    return sum(vec(-sum(y_oh .* lsm; dims=1)) .* vec(w_)) / sum(w_), st_, NamedTuple()
end

"""
    tweedie(model, ps, st, data)

Tweedie deviance loss (ρ=1.5). `data` is a tuple of `(X, y)`, `(X, y, w)`, or `(X, y, w, offset)`.
"""
function tweedie(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_,) = _handle_ensemble(pred, data[2])
    rho = eltype(data[1])(1.5)
    ep = exp.(vec(pred)); y = vec(y_)
    loss = mean(2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* ep .^ (1 - rho) / (1 - rho) .+
               ep .^ (2 - rho) / (2 - rho)))
    return loss, st_, NamedTuple()
end
function tweedie(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_) = _handle_ensemble(pred, data[2], data[3])
    rho = eltype(data[1])(1.5)
    ep = exp.(vec(pred)); y = vec(y_); w = vec(w_)
    loss = sum(w .* 2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* ep .^ (1 - rho) / (1 - rho) .+
                   ep .^ (2 - rho) / (2 - rho))) / sum(w)
    return loss, st_, NamedTuple()
end
function tweedie(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_, o_) = _handle_ensemble(pred, data[2], data[3], data[4])
    rho = eltype(data[1])(1.5)
    ep = exp.(vec(pred) .+ vec(o_)); y = vec(y_); w = vec(w_)
    loss = sum(w .* 2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* ep .^ (1 - rho) / (1 - rho) .+
                   ep .^ (2 - rho) / (2 - rho))) / sum(w)
    return loss, st_, NamedTuple()
end

"""
    gaussian_mle_loss(μ, σ, y[, w])

Negative log-likelihood for Gaussian with predicted mean `μ` and log-std `σ`.
"""
gaussian_mle_loss(μ::AbstractVector, σ::AbstractVector, y::AbstractVector) =
    -sum(-σ .- (y .- μ) .^ 2 ./ (2 .* max.(oftype.(σ, 2e-7), exp.(2 .* σ))))

gaussian_mle_loss(μ::AbstractVector, σ::AbstractVector, y::AbstractVector, w::AbstractVector) =
    -sum((-σ .- (y .- μ) .^ 2 ./ (2 .* max.(oftype.(σ, 2e-7), exp.(2 .* σ)))) .* w) / sum(w)

function gaussian_mle(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_,) = _handle_ensemble(pred, data[2])
    loss = gaussian_mle_loss(view(pred, 1, :), view(pred, 2, :), vec(y_))
    return loss, st_, NamedTuple()
end
function gaussian_mle(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_) = _handle_ensemble(pred, data[2], data[3])
    loss = gaussian_mle_loss(view(pred, 1, :), view(pred, 2, :), vec(y_), vec(w_))
    return loss, st_, NamedTuple()
end
function gaussian_mle(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred, (y_, w_, o_) = _handle_ensemble(pred, data[2], data[3], data[4])
    pred = pred .+ o_
    loss = gaussian_mle_loss(view(pred, 1, :), view(pred, 2, :), vec(y_), vec(w_))
    return loss, st_, NamedTuple()
end


get_loss_fn(::Type{<:MSE}) = mse_loss
get_loss_fn(::Type{<:MAE}) = mae_loss
get_loss_fn(::Type{<:LogLoss}) = logloss
get_loss_fn(::Type{<:MLogLoss}) = mlogloss
get_loss_fn(::Type{<:GaussianMLE}) = gaussian_mle
get_loss_fn(::Type{<:Tweedie}) = tweedie

const _loss_type_dict = Dict(
    :mse => MSE,
    :mae => MAE,
    :logloss => LogLoss,
    :tweedie => Tweedie,
    :gaussian_mle => GaussianMLE,
    :mlogloss => MLogLoss,
)

get_loss_type(loss::Symbol) = _loss_type_dict[loss]
get_loss_fn(s::Symbol) = get_loss_fn(get_loss_type(s))

end