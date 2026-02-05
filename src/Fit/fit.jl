module Fit

export fit

using ..Data
using ..Learners
using ..Models
using ..Losses
using ..Metrics

import MLJModelInterface: fit
import Reactant
import Reactant: ConcreteRArray
import Optimisers
import Optimisers: OptimiserChain, WeightDecay, Adam
import Flux
import ADTypes: AutoEnzyme

using DataFrames
using CategoricalArrays

include("callback.jl")
using .CallBacks

function init(
    config::LearnerTypes,
    df::AbstractDataFrame;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing,
)
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

    dtrain = get_df_loader_train(df; feature_names, target_name, weight_name, offset_name, batchsize, device=:cpu)

    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered)

    chain = config.arch(; nfeats, outsize)
    m = NeuroTabModel(L, chain, info)

    optim = OptimiserChain(Adam(config.lr), WeightDecay(config.wd))
    opts = Optimisers.setup(optim, m)
    
    cache = (dtrain=dtrain, loss=loss, opts=opts, info=info)
    return m, cache
end

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
    feature_names = Symbol.(feature_names)
    target_name = Symbol(target_name)
    weight_name = isnothing(weight_name) ? nothing : Symbol(weight_name)
    offset_name = isnothing(offset_name) ? nothing : Symbol(offset_name)

    m, cache = init(config, dtrain; feature_names, target_name, weight_name, offset_name)

    logger = nothing
    if !isnothing(deval)
        cb = CallBack(config, deval; feature_names, target_name, weight_name, offset_name)
        logger = init_logger(config)
        cb(logger, 0, m)
        (verbosity > 0) && @info "Init training" metric = logger[:metrics][end]
    else
        (verbosity > 0) && @info "Init training"
    end

    while m.info[:nrounds] < config.nrounds
        fit_iter!(m, cache)
        iter = m.info[:nrounds]
        if !isnothing(logger)
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

function fit_iter!(m, cache)
    loss_fn, opts, data = cache[:loss], cache[:opts], cache[:dtrain]
    
    for d in data
        x, y = d[1], d[2]
        w = length(d) >= 3 ? d[3] : nothing
        o = length(d) >= 4 ? d[4] : nothing
        
        x_ra = ConcreteRArray(x)
        y_ra = ConcreteRArray(y)
        m_ra = Reactant.to_rarray(m)
        
        # Single path - Reactant backend set via Reactant.set_default_backend()
        if !isnothing(w) && !isnothing(o)
            w_ra = ConcreteRArray(w)
            o_ra = ConcreteRArray(o)
            _, grads = Reactant.@jit Flux.withgradient(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra, w_ra, o_ra)
        elseif !isnothing(w)
            w_ra = ConcreteRArray(w)
            _, grads = Reactant.@jit Flux.withgradient(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra, w_ra)
        else
            _, grads = Reactant.@jit Flux.withgradient(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra)
        end
        
        Optimisers.update!(opts, m, grads[1])
    end
    m.info[:nrounds] += 1
    return nothing
end

end