module Infer

using ..Data
using ..Losses
using ..Models

using Flux: sigmoid, softmax!, cpu
using DataFrames: AbstractDataFrame
import MLUtils: DataLoader

export infer

"""
    infer(m::NeuroTabModel, data)

Return the inference of a `NeuroTabModel` over `data`, where `data` is `AbstractDataFrame`.
"""
function infer(m::NeuroTabModel, data::AbstractDataFrame; device=:cpu)
    m = m |> cpu  # For now, inference on CPU
    dinfer = get_df_loader_infer(data; feature_names=m.info[:feature_names], batchsize=2048, device=:cpu)
    p = infer(m, dinfer)
    return p
end


"""
    (m::NeuroTabModel)(x::AbstractMatrix)
    (m::NeuroTabModel)(data::AbstractDataFrame)

Inference for NeuroTabModel
"""
function (m::NeuroTabModel)(x::AbstractMatrix)
    p = m.chain(x)
    if size(p, 1) == 1
        p = dropdims(p; dims=1)
    end
    return p
end
function (m::NeuroTabModel)(data::AbstractDataFrame; device=:cpu)
    m = m |> cpu  # For now, inference on CPU
    dinfer = get_df_loader_infer(data; feature_names=m.info[:feature_names], batchsize=2048, device=:cpu)
    p = infer(m, dinfer)
    return p
end


function infer(m::NeuroTabModel{L}, data::DataLoader) where {L<:Union{MSE,MAE}}
    preds = Vector{Float32}[]
    for x in data
        push!(preds, Vector(m(x)))
    end
    p = vcat(preds...)
    return p
end

function infer(m::NeuroTabModel{<:LogLoss}, data::DataLoader)
    preds = Vector{Float32}[]
    for x in data
        push!(preds, Vector(m(x)))
    end
    p = vcat(preds...)
    p .= sigmoid(p)
    return p
end

function infer(m::NeuroTabModel{<:MLogLoss}, data::DataLoader)
    preds = Matrix{Float32}[]
    for x in data
        push!(preds, Matrix(m(x)'))
    end
    p = vcat(preds...)
    softmax!(p; dims=2)
    return p
end

function infer(m::NeuroTabModel{<:GaussianMLE}, data::DataLoader)
    preds = Matrix{Float32}[]
    for x in data
        push!(preds, Matrix(m(x)'))
    end
    p = vcat(preds...)
    p[:, 2] .= exp.(p[:, 2]) # reproject log(σ) into σ 
    return p
end

function infer(m::NeuroTabModel{L}, data::DataLoader) where {L<:Union{Tweedie}}
    preds = Vector{Float32}[]
    for x in data
        push!(preds, Vector(m(x)))
    end
    p = vcat(preds...)
    p .= exp.(p)
    return p
end

end # module