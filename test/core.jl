@testset "Core - internals test" begin

    learner = NeuroTabRegressor(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(
            :actA => :identity,
            :init_scale => 1.0,
            :depth => 4,
            :ntrees => 32,
            :stack_size => 1,
            :hidden_size => 1),
        loss=:mse,
        nrounds=20,
        early_stopping_rounds=2,
        batchsize=2048,
        lr=1e-2,
    )

    # stack tree
    nobs = 1_000
    nfeats = 10
    x = rand(Float32, nfeats, nobs)
    feature_names = "var_" .* string.(1:nobs)

    outsize = 1
    loss = NeuroTabModels.Losses.get_loss_fn(learner.loss)
    L = NeuroTabModels.Losses.get_loss_type(learner.loss)
    chain = learner.arch(; nfeats, outsize)
    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
    )
    m = NeuroTabModel(L, chain, info)


end

@testset "Core - Regression" begin

    Random.seed!(123)
    X, y = rand(1000, 10), randn(1000)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    target_name = "y"
    feature_names = setdiff(names(df), [target_name])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabRegressor(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(
            :depth => 3),
        loss=:mse,
        nrounds=20,
        early_stopping_rounds=2,
        lr=1e-1,
    )

    m = NeuroTabModels.fit(
        learner,
        dtrain;
        target_name,
        feature_names
    )

    m = NeuroTabModels.fit(
        learner,
        dtrain;
        target_name,
        feature_names,
        deval,
    )

end

@testset "Classification test" begin

    Random.seed!(123)
    X, y = @load_crabs
    df = DataFrame(X)
    df[!, :class] = y
    target_name = "class"
    feature_names = setdiff(names(df), [target_name])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabClassifier(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(
            :depth => 4),
        nrounds=100,
        batchsize=64,
        early_stopping_rounds=10,
        lr=3e-2,
    )

    m = NeuroTabModels.fit(
        learner,
        dtrain;
        deval,
        target_name,
        feature_names,
    )

    # Predictions depend on the number of samples in the dataset
    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) > 0.95
    @test mean(peval .== levelcode.(deval.class)) > 0.95

end
