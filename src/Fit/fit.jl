module Fit

export fit, fit_iter!

using ..Data
using ..Learners
using ..Models
using ..Losses
using ..Metrics

import Random: Xoshiro
import MLJModelInterface: fit
import Optimisers: OptimiserChain, WeightDecay, NAdam
import OneHotArrays: onehotbatch

using Lux
using Reactant
using Lux: cpu_device, reactant_device

using DataFrames
using CategoricalArrays

include("callback.jl")
using .CallBacks

function _get_device(config)
    backend = config.device == :gpu ? "gpu" : "cpu"
    Reactant.set_default_backend(backend)
    return reactant_device()
end

function _get_lux_loss(L)
    if L <: MSE
        return MSELoss()
    elseif L <: MAE
        return MAELoss()
    elseif L <: LogLoss
        return BinaryCrossEntropyLoss(; logits=Val(true))
    elseif L <: MLogLoss
        return CrossEntropyLoss(; logits=Val(true))
    else
        return MSELoss()
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
    dev = _get_device(config)

    batchsize = config.batchsize
    nfeats = length(feature_names)
    L = get_loss_type(config.loss)
    lux_loss = _get_lux_loss(L)

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

    dtrain_loader = get_df_loader_train(df; feature_names, target_name, weight_name, offset_name, batchsize)
    all_batches = collect(dtrain_loader)

    # One-hot encode targets for multiclass classification
    if L <: MLogLoss
        all_batches = map(all_batches) do batch
            x = batch[1]
            y_oh = Float32.(onehotbatch(vec(batch[2]), UInt32(1):UInt32(outsize)))
            length(batch) > 2 ? (x, y_oh, batch[3:end]...) : (x, y_oh)
        end
    end

    data = all_batches |> dev

    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered)

    chain = config.arch(; nfeats, outsize)
    m = NeuroTabModel(L, chain, info)

    rng = Xoshiro(config.seed)
    ps, st = Lux.setup(rng, m.chain) |> dev
    opt = OptimiserChain(NAdam(config.lr), WeightDecay(config.wd))
    ts = Training.TrainState(m.chain, ps, st, opt)

    cache = Dict(
        :data => data,
        :lux_loss => lux_loss,
        :train_state => ts,
    )
    return m, cache
end

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
        verbosity=1,
    )

Training function of NeuroTabModels' internal API.

# Arguments

- `config::LearnerTypes`
- `dtrain`: Must be `<:AbstractDataFrame`

# Keyword arguments

- `feature_names`:          Required kwarg, a `Vector{Symbol}` or `Vector{String}` of the feature names.
- `target_name`:            Required kwarg, a `Symbol` or `String` indicating the name of the target variable.
- `weight_name=nothing`
- `offset_name=nothing`
- `deval=nothing`:          Data for tracking evaluation metric and perform early stopping.
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

    logger = nothing
    if !isnothing(deval)
        cb = CallBack(config, deval; feature_names, target_name, weight_name, offset_name)
        logger = init_logger(config)
        _sync_params_to_model!(m, cache)
        cb(logger, 0, m)
        (verbosity > 0) && @info "Init training" metric = logger[:metrics][end]
    else
        (verbosity > 0) && @info "Init training"
    end

    while m.info[:nrounds] < config.nrounds
        fit_iter!(m, cache)
        iter = m.info[:nrounds]

        if !isnothing(logger)
            _sync_params_to_model!(m, cache)
            cb(logger, iter, m)
            if verbosity > 0 && iter % print_every_n == 0
                @info "iter $iter" metric = logger[:metrics][:metric][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        else
            (verbosity > 0 && iter % print_every_n == 0) && @info "iter $iter"
        end
    end

    _sync_params_to_model!(m, cache)
    m.info[:logger] = logger
    return m
end

function _sync_params_to_model!(m, cache)
    ts = cache[:train_state]
    cdev = cpu_device()
    m.info[:ps] = cdev(ts.parameters)
    m.info[:st] = cdev(Lux.testmode(ts.states))
end

function fit_iter!(m, cache)
    ts = cache[:train_state]
    lux_loss = cache[:lux_loss]

    for d in cache[:data]
        _, loss, _, ts = Training.single_train_step!(
            AutoEnzyme(), lux_loss, (d[1], d[2]), ts
        )
    end

    cache[:train_state] = ts
    m.info[:nrounds] += 1
    return nothing
end

end