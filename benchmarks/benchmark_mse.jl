using NeuroTabModels
using DataFrames
using BenchmarkTools
using Random: seed!
using Statistics: mean, std

Threads.nthreads()

seed!(123)
nobs = Int(1e6)
num_feat = Int(100)
@info "testing with: $nobs observations | $num_feat features."
X = rand(Float32, nobs, num_feat)
Y_raw = Float32.(X * randn(Float32, num_feat) .+ 0.1f0 * randn(Float32, nobs))
Y = Float32.((Y_raw .- mean(Y_raw)) ./ std(Y_raw))
dtrain = DataFrame(X, :auto)
feature_names = names(dtrain)
dtrain.y = Y
target_name = "y"

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
    nrounds=50,
    lr=5e-4,
    batchsize=2048,
    device=:gpu
)

@time m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval=dtrain,
    target_name,
    feature_names,
    print_every_n=1,
);

@time p_train = m(dtrain[1:10, :]; device=:cpu);
@info "Predictions (row1=μ, row2=σ):"
display(p_train)
@info "Targets:" dtrain.y[1:10]