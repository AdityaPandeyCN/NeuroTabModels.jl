module Models

export NeuroTabModel, Architecture
export NeuroTreeConfig, MLPConfig, ResNetConfig, TabMConfig
export Embeddings

using ..Losses
using Lux: Chain

abstract type Architecture end

"""
    NeuroTabModel
"""
struct NeuroTabModel{L<:LossType,C<:Chain}
    _loss_type::Type{L}
    chain::C
    info::Dict{Symbol,Any}
end

# 1. Embeddings (Base components)
include("embeddings/embeddings.jl")
using .Embeddings

# 2. Architectures
include("NeuroTree/neurotrees.jl")
using .NeuroTrees

include("TabM/TabM.jl")
using .TabM

# include("MLP/mlp.jl")
# using .MLP
# include("ResNet/resnet.jl")
# using .ResNet

end