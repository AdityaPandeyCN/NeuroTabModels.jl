using NeuroTabModels
using MLDatasets
using DataFrames
using Statistics: mean
using StatsBase: median
using CategoricalArrays
using Random

Random.seed!(123)

# -----------------------------------------------------------------------
# 1. Data Preparation
# -----------------------------------------------------------------------
println("Loading Titanic dataset...")
df = MLDatasets.Titanic().dataframe

transform!(df, :Sex => categorical => :Sex)
transform!(df, :Sex => ByRow(levelcode) => :Sex)

# NeuroTabClassifier requires target to be categorical
transform!(df, :Survived => categorical => :Survived)

transform!(df, :Age => ByRow(ismissing) => :Age_ismissing)
transform!(df, :Age => (x -> coalesce.(x, median(skipmissing(x)))) => :Age);

df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]

# Split
train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

target_name = "Survived"
feature_names = setdiff(names(df), [target_name])

# =======================================================================
# MODEL 1: TabM
# =======================================================================
println("\n[1/2] Training TabM...")

learner_tabm = NeuroTabClassifier(
    arch_name = "TabMConfig",
    arch_config = Dict(
        :k => 16,
        :n_blocks => 2,
        :d_block => 32,
        :arch_type => :tabm,
        :dropout => 0.1,
        :scaling_init => :random_signs
    ),
    loss = :logloss,
    nrounds = 100,
    early_stopping_rounds = 10,
    lr = 2e-2,
    batchsize = 64,
    device = :cpu
)

@time m_tabm = NeuroTabModels.fit(
    learner_tabm,
    dtrain;
    deval = deval,
    target_name = target_name,
    feature_names = feature_names,
    print_every_n = 20
);

# =======================================================================
# MODEL 2: NeuroTrees
# =======================================================================
println("\n[2/2] Training NeuroTrees...")

learner_tree = NeuroTabClassifier(
    arch_name = "NeuroTreeConfig",
    arch_config = Dict(
        :depth => 4,
        :ntrees => 32,
        :stack_size => 1,
        :hidden_size => 1,
        :init_scale => 1.0,
        :actA => :identity
    ),
    loss = :logloss,
    nrounds = 100,
    early_stopping_rounds = 10,
    lr = 5e-2,
    batchsize = 256,
    device = :cpu
)

@time m_tree = NeuroTabModels.fit(
    learner_tree,
    dtrain;
    deval = deval,
    target_name = target_name,
    feature_names = feature_names,
    print_every_n = 20
);

# =======================================================================
# Comparison & Evaluation
# =======================================================================
println("\n--- Final Comparison ---")

# Helper function for accuracy
function get_accuracy(model, data, target_col)
    probs = model(data) # (Batch, Classes)
    preds = [argmax(row) for row in eachrow(probs)]
    actual = Int.(levelcode.(data[!, target_col]))
    return mean(preds .== actual)
end

acc_tabm_train = get_accuracy(m_tabm, dtrain, target_name)
acc_tabm_eval  = get_accuracy(m_tabm, deval, target_name)

acc_tree_train = get_accuracy(m_tree, dtrain, target_name)
acc_tree_eval  = get_accuracy(m_tree, deval, target_name)

println("\nModel        | Train Acc | Eval Acc")
println("-------------|-----------|-----------")
println("TabM         | $(round(acc_tabm_train*100, digits=2))%    | $(round(acc_tabm_eval*100, digits=2))%")
println("NeuroTrees   | $(round(acc_tree_train*100, digits=2))%    | $(round(acc_tree_eval*100, digits=2))%")