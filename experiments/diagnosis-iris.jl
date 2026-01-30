using NeuroTabModels
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using PlotlyLight
using Random
using Flux: onecold
Random.seed!(123)

# Load Iris dataset using the correct API
features = MLDatasets.Iris.features()
labels = MLDatasets.Iris.labels()

# Create DataFrame (features is 4x150, need to transpose)
df = DataFrame(features', [:sepal_length, :sepal_width, :petal_length, :petal_width])
df.class = labels

df[!, :class] = categorical(df[!, :class])
target_name = "class"
feature_names = setdiff(names(df), [target_name])

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

depth = 3
ntrees = 4

config = NeuroTabClassifier(
    nrounds=400,
    depth=depth,
    ntrees=ntrees,
    lr=5e-2,
    batchsize=60,
    metric=:mlogloss,
    early_stopping_rounds=2,
)

m = NeuroTabModels.fit(
    config,
    dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n=10,
)

p_train = m(dtrain)
p_eval = m(deval)
println("Train accuracy: ", mean(levelcode.(dtrain[!, target_name]) .== onecold(p_train')))
println("Eval accuracy: ", mean(levelcode.(deval[!, target_name]) .== onecold(p_eval')))

nts = m.chain.layers[2]
nt = nts.trees[1]
w = nt.w

xnames = "feat" .* string.(1:4)
ynames = ["T$j/N$i" for i in 1:(2^depth-1), j in 1:ntrees]
ynames = vec(ynames)

p = plot(z=w, x=xnames, y=ynames, type="heatmap")
display(p)

println("\nDone!")