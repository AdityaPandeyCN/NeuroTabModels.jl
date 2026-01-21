using NeuroTabModels
using DataFrames
using BenchmarkTools
using Random: seed!

Threads.nthreads()

seed!(123)
nobs = Int(1e6)
num_feat = Int(100)
@info "testing with: $nobs observations | $num_feat features."
X = rand(Float32, nobs, num_feat)
Y = randn(Float32, size(X, 1))
dtrain = DataFrame(X, :auto)
feature_names = names(dtrain)
dtrain.y = Y
target_name = "y"

arch = NeuroTabModels.NeuroTreeConfig(;
    actA=:identity,
    init_scale=1.0,
    depth=4,
    ntrees=32,
    proj_size=1,
    stack_size=1,
    hidden_size=1,
)
# arch = NeuroTabModels.MLPConfig(;
#     act=:relu,
#     stack_size=1,
#     hidden_size=64,
# )

learner = NeuroTabRegressor(
    arch;
    loss=:mse,
    nrounds=20,
    early_stopping_rounds=2,
    lr=1e-2,
    device=:gpu
)

# nrounds=20: 32 sec
@time m = NeuroTabModels.fit(
    learner,
    dtrain;
    target_name,
    feature_names,
    print_every_n=10,
)

# @time p_train = m(dtrain; device=:gpu)
@time p_train = m(dtrain)
