using MLDatasets
using DataFrames
using Statistics: mean
using StatsBase: median
using CategoricalArrays
using Random
using OrderedCollections
using NeuroTabModels

Random.seed!(123)

df = MLDatasets.Titanic().dataframe

transform!(df, :Sex => categorical => :Sex)
transform!(df, :Sex => ByRow(levelcode) => :Sex)
transform!(df, :Age => ByRow(ismissing) => :Age_ismissing)
transform!(df, :Age => (x -> coalesce.(x, median(skipmissing(x)))) => :Age)
df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]
dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

target_name = "Survived"
feature_names = setdiff(names(df), ["Survived"])

configs = OrderedDict(
    # --- no embeddings ---
    "tabm" => Dict(:arch_type => :tabm, :scaling_init => :random_signs, :use_embeddings => false),
    "tabm_mini" => Dict(:arch_type => :tabm_mini, :scaling_init => :random_signs, :use_embeddings => false),
    "tabm_packed" => Dict(:arch_type => :tabm_packed, :use_embeddings => false),
    # --- with periodic embeddings ---
    "tabm+periodic" => Dict(:arch_type => :tabm, :use_embeddings => true, :embedding_type => :periodic, :d_embedding => 12),
    "tabm_mini+periodic" => Dict(:arch_type => :tabm_mini, :use_embeddings => true, :embedding_type => :periodic, :d_embedding => 12),
    "tabm_packed+periodic" => Dict(:arch_type => :tabm_packed, :use_embeddings => true, :embedding_type => :periodic, :d_embedding => 12),
    # --- with linear embeddings ---
    "tabm+linear" => Dict(:arch_type => :tabm, :use_embeddings => true, :embedding_type => :linear, :d_embedding => 12),
    # --- with piecewise embeddings ---
    "tabm+piecewise" => Dict(:arch_type => :tabm, :use_embeddings => true, :embedding_type => :piecewise, :d_embedding => 12, :n_bins => 16),
)

for (name, cfg) in configs
    @info "=" ^ 60
    @info "Testing: $name"
    @info "=" ^ 60

    arch_kwargs = Dict{Symbol,Any}(
        :k => 8, :n_blocks => 2, :d_block => 64, :dropout => 0.1,
    )
    merge!(arch_kwargs, cfg)

    arch = NeuroTabModels.TabMConfig(; arch_kwargs...)

    learner = NeuroTabRegressor(
        arch;
        loss=:logloss,
        nrounds=50,
        early_stopping_rounds=10,
        lr=1e-2,
        device=:cpu,
    )

    try
        m = NeuroTabModels.fit(
            learner, dtrain;
            deval, target_name, feature_names,
            print_every_n=25,
        )

        p_train = m(dtrain)
        p_eval = m(deval)
        train_acc = mean((p_train .> 0.5) .== (dtrain[!, target_name] .> 0.5))
        eval_acc = mean((p_eval .> 0.5) .== (deval[!, target_name] .> 0.5))
        @info "$name results" train_acc eval_acc
    catch e
        @error "$name FAILED" exception = (e, catch_backtrace())
    end
end