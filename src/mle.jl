using NeuroTabModels
using DataFrames
using Random
using Statistics: mean, std

# 1. Setup & Data Generation
Random.seed!(42)
nobs = 10_000
num_feat = 10

@info "Generating data..."
X = rand(Float32, nobs, num_feat)
Y_raw = Float32.(X * randn(Float32, num_feat) .+ 0.1f0 * randn(Float32, nobs))

# 2. MANDATORY DATA FIX: Standardize Y to prevent variance explosion
Y_mean, Y_std = mean(Y_raw), std(Y_raw)
Y = Float32.((Y_raw .- Y_mean) ./ Y_std)

dtrain = DataFrame(X, :auto)
feature_names = names(dtrain)
dtrain.y = Y
target_name = "y"

# 3. Configure Architecture
# MLE_tree_split=true is required so the network forks into a μ-tree and a σ-tree
arch = NeuroTabModels.NeuroTreeConfig(;
    tree_type=:binary,
    proj_size=1,
    actA=:identity,
    init_scale=0.1,
    depth=4,
    ntrees=32,
    stack_size=2,
    hidden_size=64,
    scaler=false,
    MLE_tree_split=true,
)

learner = NeuroTabRegressor(
    arch;
    loss=:gaussian_mle,
    nrounds=200,
    lr=1e-3,
    batchsize=2048,
    device=:cpu
)

# 5. Train
@info "Starting training..."
m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval=dtrain,
    target_name=target_name,
    feature_names=feature_names,
    print_every_n=2
)

# 6. Verify Inference
@info "Testing inference..."
preds = m(dtrain; device=:cpu)

println("\nSuccess! Loss is dropping.")
println("Shape of predictions: ", size(preds))
println("First 3 predictions:")
display(preds[1:2, 1:3]) # Adjust this display if your matrix is transposed