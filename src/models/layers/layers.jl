module Layers

export MaskedBatchNorm, CarryMask, MaskSkip
export GroupedDense, rsqrt_uniform_grouped, glorot_uniform_grouped
export AttnResidual, ResidualScale

using Lux
using Lux: StatefulLuxLayer, zeros32
using LuxCore
using LuxLib: batched_matmul
using NNlib: dot_product_attention
using Random: AbstractRNG

include("maskednorm.jl")
include("attn.jl")
include("groupeddense.jl")

"""
    _res_block(hsize, act, dropout)

ResNet residual block with masked BatchNorm so grouped pads can be ignored.
"""
function _res_block(hsize::Int, act, dropout::Float64)
    layers = Any[CarryMask(Dense(hsize => hsize)), MaskedBatchNorm(hsize, act)]
    dropout > 0 && push!(layers, CarryMask(Dropout(dropout)))
    push!(layers, CarryMask(Dense(hsize => hsize)))
    push!(layers, MaskedBatchNorm(hsize))
    return Chain(MaskSkip(Chain(layers...)), MaskedBatchNorm(hsize, act))
end

end
