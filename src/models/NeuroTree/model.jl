struct NeuroTree{DI,DP,M,P}
    ntrees::Int
    d_in::DI
    d_proj::DP
    lmask::M
    smask::M
    p::P
end
@layer NeuroTree trainable = (d_in, d_proj, p)

# function (m::NeuroTree)(x)
#     h = m.d_in(x) # [F,B] => [HNT,B]
#     h = reshape(h, size(m.d_proj.weight, 2), :) # [HNT,B] => [H,NTB]
#     nw = m.d_proj(h) # [H,NTB] => [1,NTB]
#     nw = reshape(nw, size(m.lmask, 2), :) # [1,NTB] => [N,TB]
#     lw = exp.(m.lmask * nw .- m.smask * softplus.(nw)) # [N,TB] => [L,TB]
#     lw = reshape(lw, :, size(x, 2)) # [L,TB] => [LT,B]
#     p = m.p * lw ./ m.ntrees # [P,LT] * [LT,B] => [P,B]
#     return p
# end
# function (m::NeuroTree)(x)
#     nw = m.d_in(x) # [F,B] => [NT,B]
#     nw = reshape(nw, size(m.lmask, 2), :) # [NT,B] => [N,TB]
#     lw = exp.(m.lmask * nw .- m.smask * softplus.(nw)) # [N,TB] => [L,TB]
#     lw = reshape(lw, :, size(x, 2)) # [L,TB] => [LT,B]
#     p = m.p * lw ./ m.ntrees # [P,LT] * [LT,B] => [P,B]
#     return p
# end
function (m::NeuroTree)(x)
    nw = relu.(m.d_in(x)) # [F,B] => [NT,B]
    nw = reshape(nw, size(m.lmask, 2), :) # [NT,B] => [N,TB]
    lw = exp.(m.lmask * nw .- m.smask * softplus.(nw)) # [N,TB] => [L,TB]
    lw = reshape(lw, :, size(x, 2)) # [L,TB] => [LT,B]
    p = m.p * lw ./ m.ntrees # [P,LT] * [LT,B] => [P,B]
    return p
end

"""
    NeuroTree(; ins, outs, depth=4, ntrees=64, actA=identity, init_scale=1e-1)
    NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, actA=identity, init_scale=1e-1)

Initialization of a NeuroTree.
"""
function NeuroTree(; ins, outs, tree_type=:binary, depth=4, ntrees=64, proj_size=1, actA=identity, scaler=true, init_scale=1e-1)
    lmask = get_logits_mask(Val(tree_type), depth)
    smask = get_softplus_mask(Val(tree_type), depth)
    nnodes = size(lmask, 1)
    nleaves = size(lmask, 2)

    op = NeuroTree(
        ntrees,
        Dense(ins => proj_size * nnodes * ntrees, relu), # w
        # Dense(ins => nnodes * ntrees), # w
        Dense(proj_size => 1), # s
        Float32.(lmask),
        Float32.(smask),
        Float32.(randn(outs, nleaves * ntrees) .* init_scale), # p
    )
    return op
end
function NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; tree_type=:binary, depth=4, ntrees=64, proj_size=1, actA=identity, scaler=true, init_scale=1e-1)
    lmask = get_logits_mask(Val(tree_type), depth)
    smask = get_softplus_mask(Val(tree_type), depth)
    nleaves = size(lmask, 1)
    nnodes = size(lmask, 2)

    op = NeuroTree(
        ntrees,
        Dense(ins => proj_size * nnodes * ntrees, relu), # w
        # Dense(ins => nnodes * ntrees), # w
        Dense(proj_size => 1), # s
        Float32.(lmask),
        Float32.(smask),
        Float32.(randn(outs, nleaves * ntrees) .* init_scale), # p
    )
    return op
end


"""
    get_logits_mask(::Val{:binary}, depth::Integer)
"""
function get_logits_mask(::Val{:binary}, depth::Integer)
    nodes = 2^depth - 1
    leaves = 2^depth
    mask = zeros(Bool, leaves, nodes)
    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d)
        stride = 2 * k
        for b in 1:blocks
            view(mask, (b-1)*stride+1:(b-1)*stride+k, 2^(d - 1) + b - 1) .= 1
        end
    end
    return mask
end
function get_logits_mask(::Val{:oblivious}, depth::Integer)
    leaves = 2^depth
    mask = zeros(Bool, leaves, depth)
    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d)
        stride = 2 * k
        for b in 1:blocks
            view(mask, (b-1)*stride+1:(b-1)*stride+k, d) .= 1
        end
    end
    return mask
end

"""
    get_softplus_mask(::Val{:binary}, depth::Integer)
"""
function get_softplus_mask(::Val{:binary}, depth::Integer)
    nodes = 2^depth - 1
    leaves = 2^depth
    mask = zeros(Bool, leaves, nodes)
    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d + 1)
        stride = k
        for b in 1:blocks
            view(mask, (b-1)*stride+1:(b-1)*stride+k, 2^(d - 1) + b - 1) .= 1
        end
    end
    return mask
end
function get_softplus_mask(::Val{:oblivious}, depth::Integer)
    leaves = 2^depth
    mask = ones(Bool, leaves, depth)
    return mask
end

function get_mask(::Val{:binary}, depth::Integer)
    nodes = 2^depth - 1
    leaves = 2^depth
    mask = zeros(Float32, nodes, leaves)
    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d)
        stride = 2 * k
        for b in 1:blocks
            view(mask, 2^(d - 1) + b - 1, (b-1)*stride+1:(b-1)*stride+k) .= 1
        end
    end
    return mask
end
function get_mask(::Val{:oblivious}, depth::Integer)
    leaves = 2^depth
    mask = zeros(Bool, depth, leaves)
    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d)
        stride = 2 * k
        for b in 1:blocks
            view(mask, d, (b-1)*stride+1:(b-1)*stride+k) .= true
        end
    end
    return mask
end


"""
    StackTree
A StackTree is made of a collection of NeuroTree.
"""
struct StackTree
    trees::Vector{NeuroTree}
end
@layer StackTree

function StackTree((ins, outs)::Pair{<:Integer,<:Integer}; tree_type=:binary, depth=4, ntrees=64, proj_size=1, stack_size=1, hidden_size=8, actA=identity, scaler=true, init_scale=1e-1)
    @assert stack_size == 1 || hidden_size >= outs
    trees = []
    for i in 1:stack_size
        if i == 1
            if i < stack_size
                tree = NeuroTree(ins => hidden_size; tree_type, depth, ntrees, proj_size, actA, scaler, init_scale)
                push!(trees, tree)
            else
                tree = NeuroTree(ins => outs; tree_type, depth, ntrees, proj_size, actA, scaler, init_scale)
                push!(trees, tree)
            end
        elseif i < stack_size
            tree = NeuroTree(hidden_size => hidden_size; tree_type, depth, ntrees, proj_size, actA, scaler, init_scale)
            push!(trees, tree)
        else
            tree = NeuroTree(hidden_size => outs; tree_type, depth, ntrees, proj_size, actA, scaler, init_scale)
            push!(trees, tree)
        end
    end
    m = StackTree(trees)
    return m
end

function (m::StackTree)(x::AbstractMatrix)
    p = m.trees[1](x)
    for i in 2:length(m.trees)
        if i < length(m.trees)
            p = p .+ m.trees[i](p)
        else
            _p = m.trees[i](p)
            p = view(p, 1:size(_p, 1), :) .+ _p
        end
    end
    return p
end
