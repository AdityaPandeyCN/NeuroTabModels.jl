module Fit

export fit

using ..Data
using ..Learners
using ..Models
using ..Losses
using ..Metrics

import MLJModelInterface: fit
import Reactant
import Reactant: ConcreteRArray, ConcreteRNumber
import Optimisers
import Optimisers: OptimiserChain, WeightDecay, NAdam
import Flux
import Flux: onehotbatch
import ADTypes: AutoEnzyme
import Functors: fmap

using DataFrames
using CategoricalArrays

include("callback.jl")
using .CallBacks

function _to_rarray_with_scalars(x)
    return fmap(x; exclude=v -> v isa AbstractArray{<:Number} || v isa Number) do v
        if v isa AbstractArray{<:Number}
            ConcreteRArray(v)
        elseif v isa Bool
            v
        elseif v isa Number
            ConcreteRNumber(Float32(v))
        else
            v
        end
    end
end

function _train_step!(loss_fn, m, opts, args...)
    _, grads = Flux.withgradient(loss_fn, AutoEnzyme(), m, args...)
    new_opts, new_model = Optimisers.update!(opts, m, grads[1])
    return new_opts, new_model
end

function _to_batches_ra(all_batches, L, outsize)
    if L <: MLogLoss
        return map(all_batches) do batch
            x_ra = ConcreteRArray(batch[1])
            y_onehot = Float32.(onehotbatch(batch[2], UInt32(1):UInt32(outsize)))
            y_ra = ConcreteRArray(y_onehot)
            length(batch) > 2 ? (x_ra, y_ra, map(b -> ConcreteRArray(b), batch[3:end])...) : (x_ra, y_ra)
        end
    else
        return [map(b -> ConcreteRArray(b), batch) for batch in all_batches]
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

    optim = OptimiserChain(NAdam(config.lr), WeightDecay(config.wd))
    opts = Optimisers.setup(optim, m)

    chain_ra = Reactant.to_rarray(m.chain)
    m_ra = NeuroTabModel(m._loss_type, chain_ra, Dict{Symbol,Any}())
    opts_ra = _to_rarray_with_scalars(opts)

    all_batches = collect(dtrain)
    batches_ra = _to_batches_ra(all_batches, L, outsize)
    full_batchsize = size(all_batches[1][1], ndims(all_batches[1][1]))
    is_full_batch = [size(b[1], ndims(b[1])) == full_batchsize for b in all_batches]

    compiled_step = Reactant.@compile _train_step!(loss, m_ra, opts_ra, batches_ra[1]...)

    cache = Dict(
        :loss => loss,
        :info => info,
        :compiled_step => compiled_step,
        :m_ra => m_ra,
        :opts_ra => opts_ra,
        :batches_ra => batches_ra,
        :is_full_batch => is_full_batch
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
            chain_cpu = Flux.fmap(x -> x isa ConcreteRArray ? Array(x) : x, cache[:m_ra].chain)
            Flux.loadmodel!(m.chain, chain_cpu)
            cb(logger, iter, m)
            if verbosity > 0 && iter % print_every_n == 0
                @info "iter $iter" metric = logger[:metrics][:metric][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end
    end

    chain_cpu = Flux.fmap(x -> x isa ConcreteRArray ? Array(x) : x, cache[:m_ra].chain)
    Flux.loadmodel!(m.chain, chain_cpu)
    m.info[:logger] = logger
    return m
end

function fit_iter!(m, cache)
    m_ra = cache[:m_ra]
    opts_ra = cache[:opts_ra]

    for (args_ra, is_full) in zip(cache[:batches_ra], cache[:is_full_batch])
        if is_full
            opts_ra, m_ra = cache[:compiled_step](cache[:loss], m_ra, opts_ra, args_ra...)
        else
            opts_ra, m_ra = Reactant.@jit _train_step!(cache[:loss], m_ra, opts_ra, args_ra...)
        end
    end

    cache[:opts_ra] = opts_ra
    cache[:m_ra] = m_ra
    m.info[:nrounds] += 1
    return nothing
end

end