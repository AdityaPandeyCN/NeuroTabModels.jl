module Losses

export get_loss_fn, get_loss_type
export LossType, MSE, MAE, LogLoss, MLogLoss, GaussianMLE, Tweedie

import Statistics: mean
import NNlib: logsigmoid, logsoftmax
import OneHotArrays: onehotbatch
using Lux

# -----------------------------------------------------------------------
# Loss Types
# -----------------------------------------------------------------------
abstract type LossType end
abstract type MSE <: LossType end
abstract type MAE <: LossType end
abstract type LogLoss <: LossType end
abstract type MLogLoss <: LossType end
abstract type GaussianMLE <: LossType end
abstract type Tweedie <: LossType end

# -----------------------------------------------------------------------
# 1. MSE
# -----------------------------------------------------------------------
struct MSE_Loss <: Lux.AbstractLossFunction end
(::MSE_Loss)(y_pred::AbstractArray, y) = mean((y_pred .- y) .^ 2)
(::MSE_Loss)(y_pred::AbstractArray, y, w) = sum((y_pred .- y) .^ 2 .* w) / sum(w)
(::MSE_Loss)(y_pred::AbstractArray, y, w, offset) = sum((y_pred .+ offset .- y) .^ 2 .* w) / sum(w)

# -----------------------------------------------------------------------
# 2. MAE
# -----------------------------------------------------------------------
struct MAE_Loss <: Lux.AbstractLossFunction end
(::MAE_Loss)(y_pred::AbstractArray, y) = mean(abs.(y_pred .- y))
(::MAE_Loss)(y_pred::AbstractArray, y, w) = sum(abs.(y_pred .- y) .* w) / sum(w)
(::MAE_Loss)(y_pred::AbstractArray, y, w, offset) = sum(abs.(y_pred .+ offset .- y) .* w) / sum(w)

# -----------------------------------------------------------------------
# 3. LogLoss
# -----------------------------------------------------------------------
struct Log_Loss <: Lux.AbstractLossFunction end
function (::Log_Loss)(y_pred::AbstractArray, y)
    mean((1 .- y) .* y_pred .- logsigmoid.(y_pred))
end
function (::Log_Loss)(y_pred::AbstractArray, y, w)
    sum(w .* ((1 .- y) .* y_pred .- logsigmoid.(y_pred))) / sum(w)
end
function (::Log_Loss)(y_pred::AbstractArray, y, w, offset)
    p = y_pred .+ offset
    sum(w .* ((1 .- y) .* p .- logsigmoid.(p))) / sum(w)
end

# -----------------------------------------------------------------------
# 4. MLogLoss
# -----------------------------------------------------------------------
struct MLog_Loss <: Lux.AbstractLossFunction end
function (::MLog_Loss)(y_pred::AbstractArray, y)
    k = size(y_pred, 1)
    # If y is a Vector, OneHot it. If y is a Matrix, use it as-is.
    y_oh = ndims(y) == 1 ? onehotbatch(vec(y), 1:k) : y
    p = logsoftmax(y_pred; dims=1)
    mean(-sum(y_oh .* p; dims=1))
end
function (::MLog_Loss)(y_pred::AbstractArray, y, w)
    k = size(y_pred, 1)
    y_oh = ndims(y) == 1 ? onehotbatch(vec(y), 1:k) : y
    p = logsoftmax(y_pred; dims=1)
    sum(-sum(y_oh .* p; dims=1) .* w) / sum(w)
end
function (::MLog_Loss)(y_pred::AbstractArray, y, w, offset)
    k = size(y_pred, 1)
    y_oh = ndims(y) == 1 ? onehotbatch(vec(y), 1:k) : y
    p = logsoftmax(y_pred .+ offset; dims=1)
    sum(-sum(y_oh .* p; dims=1) .* w) / sum(w)
end

# -----------------------------------------------------------------------
# 5. Tweedie
# -----------------------------------------------------------------------
struct Tweedie_Loss{T} <: Lux.AbstractLossFunction 
    rho::T
end
Tweedie_Loss() = Tweedie_Loss(1.5f0)
function (l::Tweedie_Loss)(y_pred::AbstractArray, y)
    rho = eltype(y_pred)(l.rho)
    p = exp.(y_pred)
    term1 = y .^ (2 - rho) / ((1 - rho) * (2 - rho))
    term2 = y .* p .^ (1 - rho) / (1 - rho)
    term3 = p .^ (2 - rho) / (2 - rho)
    mean(2 .* (term1 .- term2 .+ term3))
end
function (l::Tweedie_Loss)(y_pred::AbstractArray, y, w)
    rho = eltype(y_pred)(l.rho)
    p = exp.(y_pred)
    term1 = y .^ (2 - rho) / ((1 - rho) * (2 - rho))
    term2 = y .* p .^ (1 - rho) / (1 - rho)
    term3 = p .^ (2 - rho) / (2 - rho)
    sum(w .* 2 .* (term1 .- term2 .+ term3)) / sum(w)
end
function (l::Tweedie_Loss)(y_pred::AbstractArray, y, w, offset)
    rho = eltype(y_pred)(l.rho)
    p = exp.(y_pred .+ offset)
    term1 = y .^ (2 - rho) / ((1 - rho) * (2 - rho))
    term2 = y .* p .^ (1 - rho) / (1 - rho)
    term3 = p .^ (2 - rho) / (2 - rho)
    sum(w .* 2 .* (term1 .- term2 .+ term3)) / sum(w)
end

# -----------------------------------------------------------------------
# 6. Gaussian MLE
# -----------------------------------------------------------------------
struct GaussianMLE_Loss <: Lux.AbstractLossFunction end
function (::GaussianMLE_Loss)(y_pred::AbstractArray, y)
    μ = view(y_pred, 1, :)
    σ = view(y_pred, 2, :)
    T = eltype(μ)
    loss = -sum(-σ .- (y .- μ) .^ 2 ./ (2 .* max.(T(2e-7), exp.(2 .* σ))))
    return loss / length(y)
end
function (::GaussianMLE_Loss)(y_pred::AbstractArray, y, w)
    μ = view(y_pred, 1, :)
    σ = view(y_pred, 2, :)
    T = eltype(μ)
    elem_loss = -(-σ .- (y .- μ) .^ 2 ./ (2 .* max.(T(2e-7), exp.(2 .* σ))))
    sum(elem_loss .* w) / sum(w)
end
function (::GaussianMLE_Loss)(y_pred::AbstractArray, y, w, offset)
    y_adj = y_pred .+ offset
    μ = view(y_adj, 1, :)
    σ = view(y_adj, 2, :)
    T = eltype(μ)
    elem_loss = -(-σ .- (y .- μ) .^ 2 ./ (2 .* max.(T(2e-7), exp.(2 .* σ))))
    sum(elem_loss .* w) / sum(w)
end

# -----------------------------------------------------------------------
# Mappings
# -----------------------------------------------------------------------

const _loss_type_dict = Dict(
    :mse => MSE,
    :mae => MAE,
    :logloss => LogLoss,
    :tweedie => Tweedie,
    :gaussian_mle => GaussianMLE,
    :mlogloss => MLogLoss
)

get_loss_type(loss::Symbol) = _loss_type_dict[loss]

function get_loss_fn(L::Type{<:LossType})
    if L <: MSE
        return MSE_Loss()
    elseif L <: MAE
        return MAE_Loss()
    elseif L <: LogLoss
        return Log_Loss()
    elseif L <: MLogLoss
        return MLog_Loss()
    elseif L <: GaussianMLE
        return GaussianMLE_Loss()
    elseif L <: Tweedie
        return Tweedie_Loss()
    else
        return MSE_Loss()
    end
end

get_loss_fn(s::Symbol) = get_loss_fn(get_loss_type(s))

end