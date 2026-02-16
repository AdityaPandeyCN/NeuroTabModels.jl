@testset "Core - data iterators" begin

end

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

    # Test without eval data
    m = NeuroTabModels.fit(
        learner,
        dtrain;
        target_name,
        feature_names
    )

    # Test inference output
    p = m(dtrain)
    @test p isa Vector{Float32}
    @test length(p) == nrow(dtrain)
    @test all(isfinite, p)

    # Test with eval data and early stopping
    m = NeuroTabModels.fit(
        learner,
        dtrain;
        target_name,
        feature_names,
        deval,
    )

    # Test eval metric was tracked
    @test haskey(m.info, :logger)
    @test !isnothing(m.info[:logger])
    @test length(m.info[:logger][:metrics][:metric]) > 0

    # Test inference on eval data
    peval = m(deval)
    @test peval isa Vector{Float32}
    @test length(peval) == nrow(deval)
    @test all(isfinite, peval)

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

    # Test inference output shape and properties
    p_cls = m(dtrain)
    nclasses = length(levels(dtrain.class))
    @test p_cls isa Matrix{Float32}
    @test size(p_cls) == (nrow(dtrain), nclasses)
    @test all(isfinite, p_cls)
    @test all(sum(p_cls; dims=2) .≈ 1.0)  # softmax rows sum to 1

    # Test eval metric was tracked and decreased
    @test haskey(m.info, :logger)
    @test !isnothing(m.info[:logger])
    metrics = m.info[:logger][:metrics][:metric]
    @test length(metrics) > 1
    @test metrics[end] < metrics[1]  # loss decreased from init

    # Test prediction accuracy
    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) > 0.95
    @test mean(peval .== levelcode.(deval.class)) > 0.95

end