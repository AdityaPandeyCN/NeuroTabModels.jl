using NeuroTabModels
using NeuroTabModels: Losses, Models, Fit
using NeuroTabModels.Losses: get_loss_fn, get_loss_type
using NeuroTabModels.Models: NeuroTabModel
using DataFrames
using PlotlyLight
using Enzyme
using Enzyme: Duplicated, Active, Const, Reverse
import Flux: trainmode!, testmode!

#################################
# vanilla DataFrame
#################################
nobs = 1000
nfeats = 100
x, y = randn(Float32, nobs, nfeats), randn(Float32, nobs);
df = DataFrame(x, :auto);
df.y = y;
feature_names = Symbol.("x" .* string.(1:nfeats))

config = NeuroTabRegressor(;
    actA=:identity,
    depth=3,
    ntrees=32,
    init_scale=0.1,
)

loss = get_loss_fn(config.loss)
L = get_loss_type(config.loss)
chain = config.arch(; nfeats, outsize=1)
info = Dict(
    :device => :cpu,
    :nrounds => 0,
    :feature_names => feature_names
)
m = NeuroTabModel(L, chain, info)
xb = x'
yb = y

println("Forward pass:")
println(m(xb))

# Model parameters
w = m.chain.layers[2].trees[1].w
b = m.chain.layers[2].trees[1].b
p = m.chain.layers[2].trees[1].p

println("\nParameter shapes:")
println("w: ", size(w))
println("b: ", size(b))
println("p: ", size(p))

# Compute gradients with Enzyme
println("\nComputing gradients with Enzyme...")
trainmode!(m)
m(xb)  # Update BatchNorm stats
testmode!(m)

dmodel = Enzyme.make_zero(m)
ad = Enzyme.set_runtime_activity(Reverse)
Enzyme.autodiff(ad, Const(loss), Active, Duplicated(m, dmodel), Const(xb), Const(yb))

# Extract gradients from shadow model
dw = dmodel.chain.layers[2].trees[1].w
db = dmodel.chain.layers[2].trees[1].b
dp = dmodel.chain.layers[2].trees[1].p

println("\nGradient shapes:")
println("dw: ", size(dw))
println("db: ", size(db))
println("dp: ", size(dp))

println("\nGradient stats:")
println("dw - min: ", minimum(dw), " max: ", maximum(dw), " mean: ", sum(dw)/length(dw))
println("db - min: ", minimum(db), " max: ", maximum(db), " mean: ", sum(db)/length(db))
println("dp - min: ", minimum(dp), " max: ", maximum(dp), " mean: ", sum(dp)/length(dp))

# Plots
fig = plot(x=vec(w); type=:histogram)
display(fig)
fig = plot(x=vec(dw); type=:histogram)
display(fig)

fig = plot(x=vec(b); type=:histogram)
display(fig)
fig = plot(x=vec(db); type=:histogram)
display(fig)

fig = plot(x=vec(p); type=:histogram)
display(fig)
fig = plot(x=vec(dp); type=:histogram)
display(fig)

println("\nDone!")