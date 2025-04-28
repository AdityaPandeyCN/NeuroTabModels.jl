module Models

export NeuroTabModel
export NeuroTreeConfig, MLPConfig

using ..Losses
import Flux: @layer, Chain

abstract type ArchType{T} end
abstract type ChainConfig end

"""
    NeuroTabModel
"""
struct NeuroTabModel{L<:LossType,C<:Chain}
    _loss_type::Type{L}
    chain::C
    info::Dict{Symbol,Any}
end
@layer NeuroTabModel

# function get_model_chain(config; nfeats, outsize, kwargs...)
#     chain = get_model_chain(ArchType{config.model_type}, config; nfeats, outsize, kwargs...)
#     return chain
# end

include("NeuroTree/neurotrees.jl")
using .NeuroTrees
include("MLP/mlp.jl")
using .MLP

end