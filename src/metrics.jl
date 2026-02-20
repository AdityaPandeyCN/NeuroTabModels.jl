module Metrics

export metric_dict, is_maximise, get_metric

import Statistics: mean, std
import NNlib: logsigmoid, logsoftmax, softmax, relu, hardsigmoid
using Lux
using Reactant

"""
    mse(x, y; agg=mean)
    mse(x, y, w; agg=mean)
    mse(x, y, w, offset; agg=mean)
"""
function mse(m, x, y; agg=mean)
    metric = agg((vec(m(x)) .- vec(y)) .^ 2)
    return metric
end
function mse(m, x, y, w; agg=mean)
    metric = agg((vec(m(x)) .- vec(y)) .^ 2 .* vec(w))
    return metric
end
function mse(m, x, y, w, offset; agg=mean)
    metric = agg((vec(m(x)) .+ vec(offset) .- vec(y)) .^ 2 .* vec(w))
    return metric
end

"""
    mae(x, y; agg=mean)
    mae(x, y, w; agg=mean)
    mae(x, y, w, offset; agg=mean)
"""
function mae(m, x, y; agg=mean)
    metric = agg(abs.(vec(m(x)) .- vec(y)))
    return metric
end
function mae(m, x, y, w; agg=mean)
    metric = agg(abs.(vec(m(x)) .- vec(y)) .* vec(w))
    return metric
end
function mae(m, x, y, w, offset; agg=mean)
    metric = agg(abs.(vec(m(x)) .+ vec(offset) .- vec(y)) .* vec(w))
    return metric
end


"""
    logloss(x, y; agg=mean)
    logloss(x, y, w; agg=mean)
    logloss(x, y, w, offset; agg=mean)
"""
function logloss(m, x, y; agg=mean)
    p = vec(m(x))
    metric = agg((1 .- vec(y)) .* p .- logsigmoid.(p))
    return metric
end
function logloss(m, x, y, w; agg=mean)
    p = vec(m(x))
    metric = agg(((1 .- vec(y)) .* p .- logsigmoid.(p)) .* vec(w))
    return metric
end
function logloss(m, x, y, w, offset; agg=mean)
    p = vec(m(x)) .+ vec(offset)
    metric = agg(((1 .- vec(y)) .* p .- logsigmoid.(p)) .* vec(w))
    return metric
end


"""
    tweedie(x, y; agg=mean)
    tweedie(x, y, w; agg=mean)
    tweedie(x, y, w, offset; agg=mean)
"""
function tweedie(m, x, y; agg=mean)
    rho = eltype(x)(1.5)
    p = exp.(vec(m(x)))
    agg(2 .* (vec(y) .^ (2 - rho) / (1 - rho) / (2 - rho) - vec(y) .* p .^ (1 - rho) / (1 - rho) +
              p .^ (2 - rho) / (2 - rho))
    )
end
function tweedie(m, x, y, w)
    agg = mean
    rho = eltype(x)(1.5)
    p = exp.(vec(m(x)))
    agg(vec(w) .* 2 .* (vec(y) .^ (2 - rho) / (1 - rho) / (2 - rho) - vec(y) .* p .^ (1 - rho) / (1 - rho) +
                   p .^ (2 - rho) / (2 - rho))
    )
end
function tweedie(m, x, y, w, offset; agg=mean)
    rho = eltype(x)(1.5)
    p = exp.(vec(m(x)) .+ vec(offset))
    agg(vec(w) .* 2 .* (vec(y) .^ (2 - rho) / (1 - rho) / (2 - rho) - vec(y) .* p .^ (1 - rho) / (1 - rho) +
                   p .^ (2 - rho) / (2 - rho))
    )
end

"""
    mlogloss(x, y; agg=mean)
    mlogloss(x, y, w; agg=mean)
    mlogloss(x, y, w, offset; agg=mean)
"""
function mlogloss(m, x, y; agg=mean)
    p = logsoftmax(m(x); dims=1)
    k = size(p, 1)
    raw = vec(-sum(((UInt32(1):UInt32(k)) .== reshape(y, 1, :)) .* p; dims=1))
    metric = agg(raw)
    return metric
end
function mlogloss(m, x, y, w; agg=mean)
    p = logsoftmax(m(x); dims=1)
    k = size(p, 1)
    raw = vec(-sum(((UInt32(1):UInt32(k)) .== reshape(y, 1, :)) .* p; dims=1))
    metric = agg(raw .* vec(w))
    return metric
end
function mlogloss(m, x, y, w, offset; agg=mean)
    p = logsoftmax(m(x) .+ offset; dims=1)
    k = size(p, 1)
    raw = vec(-sum(((UInt32(1):UInt32(k)) .== reshape(y, 1, :)) .* p; dims=1))
    metric = agg(raw .* vec(w))
    return metric
end


gaussian_loss_elt(μ, σ, y) = -σ - (y - μ)^2 / (2 * max(2.0f-7, exp(2 * σ)))


""""
    gaussian_mle(x, y; agg=mean)
    gaussian_mle(x, y, w; agg=mean)
    gaussian_mle(x, y, w, offset; agg=mean)
"""
function gaussian_mle(m, x, y; agg=mean)
    p = m(x)
    metric = agg(gaussian_loss_elt.(view(p, 1, :), view(p, 2, :), vec(y)))
    return metric
end
function gaussian_mle(m, x, y, w; agg=mean)
    p = m(x)
    metric = agg(gaussian_loss_elt.(view(p, 1, :), view(p, 2, :), vec(y)) .* vec(w))
    return metric
end
function gaussian_mle(m, x, y, w, offset; agg=mean)
    p = m(x) .+ offset
    metric = agg(gaussian_loss_elt.(view(p, 1, :), view(p, 2, :), vec(y)) .* vec(w))
    return metric
end

function get_metric(ts::Training.TrainState, f::Function, data)
    ps, st = ts.parameters, Lux.testmode(ts.states)
    model_compiled = @compile ts.model(first(data)[1], ps, st)
    m = x -> first(model_compiled(x, ps, st))
    
    metric = 0.0f0
    ws = 0.0f0
    for d in data
        metric += f(m, d...; agg=sum)
        if length(d) >= 3
            ws += sum(d[3])
        else
            ws += size(d[2], ndims(d[2]))
        end
    end
    metric = metric / ws
    return metric
end

const metric_dict = Dict(
    :mse => mse,
    :mae => mae,
    :logloss => logloss,
    :mlogloss => mlogloss,
    :gaussian_mle => gaussian_mle,
    :tweedie => tweedie,
)

is_maximise(::typeof(mse)) = false
is_maximise(::typeof(mae)) = false
is_maximise(::typeof(logloss)) = false
is_maximise(::typeof(mlogloss)) = false
is_maximise(::typeof(gaussian_mle)) = true
is_maximise(::typeof(tweedie)) = false

end