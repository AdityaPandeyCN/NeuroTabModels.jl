module Fit

export fit

using ..Data
using ..Learners
using ..Models
using ..Losses
using ..Metrics

import Random: Xoshiro
import MLJModelInterface: fit
import CUDA, cuDNN
import Optimisers
import Optimisers: OptimiserChain, WeightDecay, Adam, NAdam, Nesterov, Descent, Momentum, AdaDelta

using Lux
using Enzyme, Reactant

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

    Reactant.set_default_backend("gpu")
    dev = reactant_device()
    @info "dev" dev
    # dev = gpu_device()
    # dev = cpu_device()
    data = get_df_loader_train(df; feature_names, target_name, weight_name, offset_name, batchsize, device) |> dev

    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered)

    chain = config.arch(; nfeats, outsize)
    m = NeuroTabModel(L, chain, info)

    # Parameter and State Variables
    rng = Xoshiro(config.seed)
    ps, st = Lux.setup(rng, m.chain) |> dev
    opt = OptimiserChain(NAdam(config.lr), WeightDecay(config.wd))

    cache = (data=data, ps=ps, st=st, opt=opt, info=info)
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
        device=:cpu,
        gpuID=0,
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

    ts = Training.TrainState(m.chain, cache[:ps], cache[:st], cache[:opt])
    while m.info[:nrounds] < config.nrounds
        for d in cache[:data]
            gs, loss, stats, ts = Training.single_train_step!(
                AutoEnzyme(),
                MSELoss(),
                # BinaryCrossEntropyLoss(; logits=Val(true)),
                (d[1], d[2]),
                ts
            )
        end
        m.info[:nrounds] += 1

        iter = m.info[:nrounds]
        if !isnothing(logger)
            cb(logger, iter, m)
            if verbosity > 0 && iter % print_every_n == 0
                @info "iter $iter" metric = logger[:metrics][:metric][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        else
            (verbosity > 0 && iter % print_every_n == 0) && @info "iter $iter"
        end
    end
    m.info[:logger] = logger
    return m
end

# function fit_iter!(m, ts, data)
#     # ts, data = cache[:ts], cache[:data]
#     for d in data
#         gs, loss, stats, ts = Training.single_train_step!(
#             AutoEnzyme(),
#             MSELoss(),
#             (d[1], d[2]),
#             ts
#         )
#     end
#     m.info[:nrounds] += 1
#     return nothing
# end

end