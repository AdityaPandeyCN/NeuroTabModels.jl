module Infer

using ..Data
using ..Losses
using ..Models

using Lux
using Lux: cpu_device, reactant_device
using Reactant
using NNlib: sigmoid, softmax!
using DataFrames: AbstractDataFrame
import MLUtils: DataLoader

export infer

function _get_device(device::Symbol)
    if device == :gpu
        Reactant.set_default_backend("gpu")
        return reactant_device()
    else
        return cpu_device()
    end
end

"""
    _setup_infer(m, data, device) -> (ps, st, model_fn, batches)

"""
function _setup_infer(m::NeuroTabModel, data, device::Symbol)
    dev = _get_device(device)
    ps = dev(m.info[:ps])
    st = dev(m.info[:st])
    batches = collect(data) |> dev

    if device == :gpu
        compiled_model = Reactant.@compile m.chain(first(batches), ps, st)
        return ps, st, compiled_model, batches
    else
        return ps, st, m.chain, batches
    end
end

# Run single forward pass on device, pull result back to CPU
function _forward_batch(model_fn, x, ps, st)
    y, _ = model_fn(x, ps, st)
    return cpu_device()(y)
end

function _postprocess(::Type{<:Union{MSE,MAE}}, raw_preds::Vector)
    return vcat([vec(p) for p in raw_preds]...)
end
function _postprocess(::Type{<:LogLoss}, raw_preds::Vector)
    p = vcat([vec(p) for p in raw_preds]...)
    return sigmoid.(p)
end
function _postprocess(::Type{<:MLogLoss}, raw_preds::Vector)
    # Transpose from [class, batch] -> [batch, class], then softmax
    p = vcat([Matrix(p') for p in raw_preds]...)
    softmax!(p; dims=2)
    return p
end
function _postprocess(::Type{<:GaussianMLE}, raw_preds::Vector)
    # Transpose, then exponentiate the σ column (log(σ) -> σ)
    p = vcat([Matrix(p') for p in raw_preds]...)
    p[:, 2] .= exp.(p[:, 2])
    return p
end
function _postprocess(::Type{<:Tweedie}, raw_preds::Vector)
    p = vcat([vec(p) for p in raw_preds]...)
    return exp.(p)
end

"""
    infer(m::NeuroTabModel, data::DataLoader; device=:cpu)

"""
function infer(m::NeuroTabModel{L}, data::DataLoader; device=:cpu) where {L}
    ps, st, model_fn, batches = _setup_infer(m, data, device)
    raw_preds = [_forward_batch(model_fn, x, ps, st) for x in batches]
    return _postprocess(L, raw_preds)
end

"""
    infer(m::NeuroTabModel, data::AbstractDataFrame; device=:cpu)

"""
function infer(m::NeuroTabModel, data::AbstractDataFrame; device=:cpu)
    dinfer = get_df_loader_infer(data; feature_names=m.info[:feature_names], batchsize=2048)
    return infer(m, dinfer; device=device)
end

"""
    (m::NeuroTabModel)(x::AbstractMatrix; device=:cpu)

"""
function (m::NeuroTabModel{L})(x::AbstractMatrix; device=:cpu) where {L}
    dev = _get_device(device)
    ps, st = dev(m.info[:ps]), dev(m.info[:st])
    x_dev = dev(x)
    if device == :gpu
        model_fn = Reactant.@compile m.chain(x_dev, ps, st)
    else
        model_fn = m.chain
    end
    raw_pred = _forward_batch(model_fn, x_dev, ps, st)
    return _postprocess(L, [raw_pred])
end

"""
    (m::NeuroTabModel)(data::AbstractDataFrame; device=:cpu)

"""
function (m::NeuroTabModel)(data::AbstractDataFrame; device=:cpu)
    return infer(m, data; device=device)
end

end # module