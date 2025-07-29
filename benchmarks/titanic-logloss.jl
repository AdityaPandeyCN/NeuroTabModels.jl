using MLDatasets
using DataFrames
using Statistics: mean
using StatsBase: median
using CategoricalArrays
using Random
using CategoricalArrays
using EvoCore.IOTools
using OrderedCollections
using NeuroTabModels

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

arch = NeuroTabModels.NeuroTreeConfig(;
    actA=:identity,
    init_scale=1.0,
    depth=4,
    ntrees=32,
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
    loss=:logloss,
    nrounds=400,
    early_stopping_rounds=2,
    lr=1e-2,
)

# # TODO: move in Modeler
# function load_hyper_list(path::String)
#     js = load_json(path, "local")
#     hyper_list = Vector{OrderedDict{Symbol,Any}}(js)
#     return hyper_list
# end
# hyper_list = [Dict(:arch_name => "NeuroTreeConfig", :arch_config => Dict(:actA => "identity", :depth => 4), :loss => :logloss, :lr => 0.03)]
# save_json(hyper_list, joinpath(@__DIR__, "hyper.json"), "local")
# hyper_list = load_hyper_list(joinpath(@__DIR__, "hyper.json"))
# hyper = hyper_list[1]

# typeof(hyper[:arch_config]) <: AbstractDict
# regressor = Models.NeuroTreeConfig
# learner = regressor(; hyper[:arch_config]...)
# fieldnames(regressor)
# fieldnames(NeuroTabRegressor)
# learner = NeuroTabRegressor(; hyper...)

m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n=10,
)

p_train = m(dtrain)
p_eval = m(deval)

@info mean((p_train .> 0.5) .== (dtrain[!, target_name] .> 0.5))
@info mean((p_eval .> 0.5) .== (deval[!, target_name] .> 0.5))
