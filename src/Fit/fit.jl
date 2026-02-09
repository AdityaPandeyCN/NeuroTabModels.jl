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

function _compile_grad_fn(loss_fn, m, batch, has_w, has_o)
    m_ra = Reactant.to_rarray(m)
    x_ra = ConcreteRArray(batch[1])
    y_ra = ConcreteRArray(batch[2])

    grad_fn = if has_w && has_o
        Reactant.@compile Flux.withgradient(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra,
            ConcreteRArray(batch[3]), ConcreteRArray(batch[4]))
    elseif has_w
        Reactant.@compile Flux.withgradient(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra,
            ConcreteRArray(batch[3]))
    else
        Reactant.@compile Flux.withgradient(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra)
    end
    full_batchsize = size(batch[1], ndims(batch[1]))
    return grad_fn, full_batchsize
end

function _grad_step(grad_fn, loss_fn, m_ra, x_ra, y_ra, d, has_w, has_o, is_full)
    if has_w && has_o
        w_ra, o_ra = ConcreteRArray(d[3]), ConcreteRArray(d[4])
        return is_full ?
            grad_fn(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra, w_ra, o_ra) :
            Reactant.@jit Flux.withgradient(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra, w_ra, o_ra)
    elseif has_w
        w_ra = ConcreteRArray(d[3])
        return is_full ?
            grad_fn(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra, w_ra) :
            Reactant.@jit Flux.withgradient(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra, w_ra)
    else
        return is_full ?
            grad_fn(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra) :
            Reactant.@jit Flux.withgradient(loss_fn, AutoEnzyme(), m_ra, x_ra, y_ra)
    end
end

function _to_cpu_grads(grads)
    return Flux.fmap(grads) do x
        x isa Reactant.ConcreteRArray ? Array(x) : x
    end
end

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

    dtrain = get_df_loader_train(df; feature_names, target_name, weight_name, offset_name, batchsize)

    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered)

    chain = config.arch(; nfeats, outsize)
    m = NeuroTabModel(L, chain, info)

    optim = OptimiserChain(Adam(config.lr), WeightDecay(config.wd))
    opts = Optimisers.setup(optim, m)

    # Compile gradient function once using first batch as template
    first_batch = first(dtrain)
    has_w = length(first_batch) >= 3
    has_o = length(first_batch) >= 4
    grad_fn, full_batchsize = _compile_grad_fn(loss, m, first_batch, has_w, has_o)

    cache = (
        dtrain=dtrain, loss=loss, opts=opts, info=info,
        grad_fn=grad_fn, full_batchsize=full_batchsize,
        has_w=has_w, has_o=has_o,
    )
    return m, cache
end


"""
    function fit(
        config::NeuroTypes,
        dtrain;
        feature_names,
        target_name,
        weight_name=nothing,
        offset_name=nothing,
        deval=nothing,
        metric=nothing,
        print_every_n=9999,
        early_stopping_rounds=9999,
        verbosity=1,
    )

Training function of NeuroTabModels' internal API.

# Arguments

- `config::LearnerTypes`
- `dtrain`: Must be `<:AbstractDataFrame`  

# Keyword arguments

- `feature_names`:          Required kwarg, a `Vector{Symbol}` or `Vector{String}` of the feature names.
- `target_name`             Required kwarg, a `Symbol` or `String` indicating the name of the target variable.  
- `weight_name=nothing`
- `offset_name=nothing`
- `deval=nothing`           Data for tracking evaluation metric and perform early stopping.
- `print_every_n=9999`
- `verbosity=1`
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
    loss_fn = cache[:loss]
    opts = cache[:opts]
    data = cache[:dtrain]
    grad_fn = cache[:grad_fn]
    full_batchsize = cache[:full_batchsize]
    has_w = cache[:has_w]
    has_o = cache[:has_o]

    for d in data
        m_ra = Reactant.to_rarray(m)
        x_ra = ConcreteRArray(d[1])
        y_ra = ConcreteRArray(d[2])

        is_full = size(d[1], ndims(d[1])) == full_batchsize
        _, grads = _grad_step(grad_fn, loss_fn, m_ra, x_ra, y_ra, d, has_w, has_o, is_full)

        Optimisers.update!(opts, m, _to_cpu_grads(grads[1]))
    end
    m.info[:nrounds] += 1
    return nothing
end

end