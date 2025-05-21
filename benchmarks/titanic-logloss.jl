using MLDatasets
using DataFrames
using Statistics: mean
using StatsBase: median
using CategoricalArrays
using Random
using CategoricalArrays
# using CUDA
using NeuroTabModels
using NeuroTabModels: Models

Random.seed!(123)

df = MLDatasets.Titanic().dataframe

# convert string feature to Categorical
transform!(df, :Sex => categorical => :Sex)
transform!(df, :Sex => ByRow(levelcode) => :Sex)

# treat string feature and missing values
transform!(df, :Age => ByRow(ismissing) => :Age_ismissing)
transform!(df, :Age => (x -> coalesce.(x, median(skipmissing(x)))) => :Age);

# remove unneeded variables
df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

target_name = "Survived"
feature_names = setdiff(names(df), ["Survived"])

arch = Models.NeuroTreeConfig(;
    actA=:identity,
    init_scale=1.0,
    depth=4,
    ntrees=32,
    stack_size=1,
    hidden_size=1,
)
arch = Models.MLPConfig(;
    act=:relu,
    stack_size=1,
    hidden_size=64,
)

learner = NeuroTabRegressor(;
    arch,
    loss=:logloss,
    nrounds=400,
    lr=1e-3,
)

m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval,
    target_name,
    feature_names,
    metric=:logloss,
    print_every_n=10,
    early_stopping_rounds=2,
    device=:cpu
)

p_train = m(dtrain)
p_eval = m(deval)

@info mean((p_train .> 0.5) .== (dtrain[!, target_name] .> 0.5))
@info mean((p_eval .> 0.5) .== (deval[!, target_name] .> 0.5))
