module Fit

export fit

using ..Data
using ..Learners
using ..Models
using ..Losses
using ..Metrics

import MLJModelInterface: fit
import CUDA, cuDNN
import Enzyme
import Enzyme: Duplicated, Active, Const, Reverse
import Optimisers
import Optimisers: OptimiserChain, WeightDecay, Adam, NAdam, Nesterov, Descent, Momentum, AdaDelta
import Flux: trainmode!, testmode!, cpu, gpu

using DataFrames
using CategoricalArrays

include("callback.jl")
using .CallBacks

# Configure Enzyme once at module load
function __init__()
    Enzyme.API.strictAliasing!(false)
end

function init(
    config::LearnerTypes,
    df::AbstractDataFrame;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing,
)

    device = config.device
    batchsize = config.batchsize
    nfeats = length(feature_names)
    loss = get_loss_fn(config.loss)
    L = get_loss_type(config.loss)

    target_levels = nothing
    target_isordered = false
    outsize = 1
    if L <: MLogLoss
        eltype(df[!, target_name]) <: CategoricalValue || error("Target variable `$target_name` must have its elements `<: CategoricalValue`")
        target_levels = CategoricalArrays.levels(df[!, target_name])
        target_isordered = isordered(df[!, target_name])
        outsize = length(target_levels)
    elseif L <: GaussianMLE
        outsize = 2
    end

    dtrain = get_df_loader_train(df; feature_names, target_name, weight_name, offset_name, batchsize, device)

    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered)

    chain = config.arch(; nfeats, outsize)
    m = NeuroTabModel(L, chain, info)
    if device == :gpu
        m = m |> gpu
    end

    optim = OptimiserChain(NAdam(config.lr), WeightDecay(config.wd))
    opts = Optimisers.setup(optim, m)

    cache = (dtrain=dtrain, loss=loss, opts=opts, info=info)
    return m, cache
end


"""
    function fit(...)

Training function of NeuroTabModels' internal API.
"""
function fit(
    config::LearnerTypes,
    dtrain;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing,
    deval=nothing,
    print_every_n=9999,
    verbosity=1
)

    device = Symbol(config.device)
    if device == :gpu
        CUDA.device!(config.gpuID)
    end

    feature_names = Symbol.(feature_names)
    target_name = Symbol(target_name)
    weight_name = isnothing(weight_name) ? nothing : Symbol(weight_name)
    offset_name = isnothing(offset_name) ? nothing : Symbol(offset_name)

    m, cache = init(config, dtrain; feature_names, target_name, weight_name, offset_name)

    # initialize callback and logger if tracking eval data
    logger = nothing
    if !isnothing(deval)
        cb = CallBack(config, deval; feature_names, target_name, weight_name, offset_name)
        logger = init_logger(config)
        testmode!(m)
        cb(logger, 0, m)
        (verbosity > 0) && @info "Init training" metric = logger[:metrics][end]
    else
        (verbosity > 0) && @info "Init training"
    end

    while m.info[:nrounds] < config.nrounds
        fit_iter!(m, cache)
        iter = m.info[:nrounds]
        if !isnothing(logger)
            testmode!(m)
            cb(logger, iter, m)
            if verbosity > 0 && iter % print_every_n == 0
                @info "iter $iter" metric = logger[:metrics][:metric][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end
    end

    m.info[:logger] = logger
    return m
end

# Enzyme-based gradient computation
# Uses runtime activity mode to handle BatchNorm's dynamic allocations
function compute_grads(loss, model, d::Tuple)
    # Update BatchNorm running stats in train mode (outside autodiff)
    trainmode!(model)
    model(d[1])
    
    # Switch to test mode for gradient computation (avoids mutation issues)
    testmode!(model)
    
    # Create shadow model for gradient accumulation
    dmodel = Enzyme.make_zero(model)
    
    # Use runtime activity mode to handle dynamic allocations in BatchNorm
    ad = Enzyme.set_runtime_activity(Reverse)
    
    # Dispatch based on batch tuple size
    if length(d) == 2
        x, y = d
        Enzyme.autodiff(ad, Const(loss), Active, Duplicated(model, dmodel), Const(x), Const(y))
    elseif length(d) == 3
        x, y, w = d
        Enzyme.autodiff(ad, Const(loss), Active, Duplicated(model, dmodel), Const(x), Const(y), Const(w))
    elseif length(d) == 4
        x, y, w, o = d
        Enzyme.autodiff(ad, Const(loss), Active, Duplicated(model, dmodel), Const(x), Const(y), Const(w), Const(o))
    else
        error("Unexpected batch tuple length: $(length(d))")
    end
    
    return dmodel
end

function fit_iter!(m, cache)
    loss, opts, data = cache[:loss], cache[:opts], cache[:dtrain]
    
    # Memory cleanup
    GC.gc(true)
    if typeof(cache[:dtrain]) <: CUDA.CuIterator
        CUDA.reclaim()
    end
    
    for d in data
        grads = compute_grads(loss, m, d)
        Optimisers.update!(opts, m, grads)
    end
    
    m.info[:nrounds] += 1
    return nothing
end

end