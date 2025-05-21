module Models

export NeuroTabModel, Architecture
export NeuroTreeConfig, MLPConfig

using ..Losses
import Flux: @layer, Chain

abstract type Architecture end

"""
    NeuroTabModel
"""
struct NeuroTabModel{L<:LossType,C<:Chain}
    _loss_type::Type{L}
    chain::C
    info::Dict{Symbol,Any}
end
@layer NeuroTabModel

include("NeuroTree/neurotrees.jl")
using .NeuroTrees
include("MLP/mlp.jl")
using .MLP

end