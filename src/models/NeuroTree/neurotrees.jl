module NeuroTrees

export NeuroTreeConfig

using Random
using Lux
using LuxCore
using NNlib: softplus, sigmoid, sigmoid_fast, hardsigmoid, tanh_fast, hardtanh

import ..Losses: get_loss_type, GaussianMLE
import ..Models: Architecture

include("model.jl")

struct NeuroTreeConfig <: Architecture
    tree_type::Symbol
    actA::Symbol
    depth::Int
    ntrees::Int
    proj_size::Int
    hidden_size::Int
    stack_size::Int
    scaler::Bool
    init_scale::Float32
    MLE_tree_split::Bool
end

function NeuroTreeConfig(; kwargs...)
    args = Dict{Symbol,Any}(
        :tree_type => :binary,
        :actA => :identity,
        :depth => 4,
        :ntrees => 32,
        :proj_size => 1,
        :hidden_size => 1,
        :stack_size => 1,
        :scaler => true,
        :init_scale => 0.1,
        :MLE_tree_split => false,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @warn "Following $(length(args_ignored)) provided arguments will be ignored: $(join(args_ignored, ", "))."

    args_default = setdiff(keys(args), keys(kwargs))
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(join(args_default, ", "))."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    config = NeuroTreeConfig(
        Symbol(args[:tree_type]),
        Symbol(args[:actA]),
        args[:depth],
        args[:ntrees],
        args[:proj_size],
        args[:hidden_size],
        args[:stack_size],
        args[:scaler],
        args[:init_scale],
        args[:MLE_tree_split],
    )

    return config
end

function (config::NeuroTreeConfig)(; nfeats, outsize)
    function build_block(n_in, n_out)
        create_tree() = NeuroTree(n_in => n_out;
            tree_type=config.tree_type,
            depth=config.depth,
            trees=config.ntrees, 
            actA=act_dict[config.actA],
            scaler=config.scaler,
            init_scale=config.init_scale
        )

        if config.stack_size == 1
            return create_tree()
        else
            return Parallel(+, [create_tree() for _ in 1:config.stack_size]...)
        end
    end

    if config.MLE_tree_split && outsize == 2
        outsize ÷= 2
        chain = Chain(
            BatchNorm(nfeats),
            Parallel(
                vcat,
                build_block(nfeats, outsize),
                build_block(nfeats, outsize),
            )
        )
    else
        chain = Chain(
            BatchNorm(nfeats),
            build_block(nfeats, outsize)
        )
    end
    
    return chain
end

function _identity_act(x)
    return x ./ sum(abs.(x), dims=2)
end
function _tanh_act(x)
    x = tanh_fast.(x)
    return x ./ sum(abs.(x), dims=2)
end
function _hardtanh_act(x)
    x = hardtanh.(x)
    return x ./ sum(abs.(x), dims=2)
end

const act_dict = Dict(
    :identity => _identity_act,
    :tanh => _tanh_act,
    :hardtanh => _hardtanh_act,
)

end