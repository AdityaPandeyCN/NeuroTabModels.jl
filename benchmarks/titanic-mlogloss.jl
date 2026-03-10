using NeuroTabModels
using MLDatasets
using DataFrames
using Statistics: mean
using StatsBase: median
using CategoricalArrays
using Random
using CUDA
using CategoricalArrays

Random.seed!(123)

df = MLDatasets.Titanic().dataframe

# convert target variable to a categorical
transform!(df, :Survived => categorical => :y_cat)

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

target_name = "y_cat"
feature_names = setdiff(names(df), ["y_cat", "Survived"])
eltype(dtrain[:, "y_cat"])

arch = NeuroTabModels.NeuroTreeConfig(;
    tree_type=:binary,
    proj_size=1,
    init_scale=0.1,
    depth=4,
    ntrees=16,
    stack_size=1,
    hidden_size=1,
    actA=:identity,
    MLE_tree_split=false,
)
# arch = NeuroTabModels.TabMConfig(;
#     arch_type=:tabm,
#     k=32,
#     d_block=64,
#     n_blocks=3,
#     dropout=0.1,
#     bins=nothing,
#     use_embeddings=false,
#     embedding_type=:periodic,
#     d_embedding=16,
#     scaling_init=:random_signs,
# )

learner = NeuroTabClassifier(
    arch;
    nrounds=100,
    early_stopping_rounds=5,
    lr=1e-2,
    device=:cpu
)

@time m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n=1,
);

p_train = m(dtrain)
p_train_idx = [argmax(p) for p in eachrow(p_train)]

p_eval = m(deval)
p_eval_idx = [argmax(p) for p in eachrow(p_eval)]

@info mean(p_train_idx .== levelcode.(dtrain[!, target_name]))
@info mean(p_eval_idx .== levelcode.(deval[!, target_name]))
