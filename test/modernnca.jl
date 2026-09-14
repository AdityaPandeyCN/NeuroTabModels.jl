# Dense reference: full softmax over all keys, no chunking.
function dense_nca(model, zq, (z3, zt), y; mask_self=false)
    modernnca = NeuroTabModels.Models.ModernNCA
    zk = hcat(reshape(z3, size(z3, 1), :), zt)
    _, s = modernnca._scores(model, zq, zk; mask_self)
    α = exp.(s .- maximum(s; dims=1))
    α = α ./ sum(α; dims=1)
    return modernnca._finalize(model.loss, modernnca._train_targets(model, y) * α)
end

@testset "ModernNCA integration" begin
    rng = Xoshiro(123)
    x = randn(rng, Float32, 40, 4)
    df = DataFrame(x, :auto)
    df.y = x[:, 1] .- 0.5f0 .* x[:, 2]
    dtrain, deval = df[1:32, :], df[33:end, :]
    features = names(df, r"x")

    arch = NeuroTabModels.ModernNCAConfig(;
        d_embedding=8, n_blocks=0, sample_rate=0.5,
        corpus_chunk_size=7)
    learner = NeuroTabRegressor(arch;
        embedding_config=NeuroTabModels.LinearEmbeddings(; d_embedding=2),
        nrounds=5, early_stopping_rounds=5, batchsize=16,
        scale_target=false, backend=:zygote, device=:cpu)
    model = NeuroTabModels.fit(
        learner, dtrain;
        feature_names=features, target_name=:y, deval, verbosity=0)

    @test size(model.info[:nca_ref].cx, 2) == nrow(dtrain)
    prediction = model(deval)
    @test all(isfinite, prediction)
    metrics = model.info[:logger][:metrics][:metric]
    @test length(unique(metrics)) > 1
    @test sum(abs2, prediction .- deval.y) / nrow(deval) ≈
          last(metrics) rtol=1f-5

    deval.group = repeat(1:2; inner=4)
    model.info[:group_name] = :group
    @test length(model(deval)) == nrow(deval)
    select!(deval, Not(:group))

    arch = NeuroTabModels.ModernNCAConfig(; d_embedding=8, n_blocks=0)
    @test_throws ArgumentError arch(
        ; ins=4, outsize=2,
        loss=NeuroTabModels.Losses.GaussianMLE())
    @test_throws ArgumentError NeuroTabModels.ModernNCAConfig(
        ; corpus_chunk_size=0)

    model = arch(
        ; ins=4, outsize=1, loss=NeuroTabModels.Losses.MSE())
    @test_throws ArgumentError NeuroTabModels.Models.train_dataloader(
        arch, NeuroTabModels.Models.NeuroTabModel(
            NeuroTabModels.Losses.MSE(), model, Dict{Symbol,Any}()),
        nothing, DataFrame(randn(Float32, 3, 4), :auto);
        feature_names=[:x1, :x2, :x3, :x4], target_name=:x1,
        loss=NeuroTabModels.Losses.MSE(), scalers=nothing, batchsize=2,
        dev=identity, rng=Xoshiro(0), backend=:enzyme)

    fitted = NeuroTabModels.Models.NeuroTabModel(
        NeuroTabModels.Losses.MSE(), model, Dict{Symbol,Any}())
    for unsupported in
        ((; weight_name=:w), (; offset_name=:o), (; group_name=:g))
        @test_throws ArgumentError NeuroTabModels.Models.train_dataloader(
            arch, fitted, nothing, dtrain;
            feature_names=features, target_name=:y,
            loss=NeuroTabModels.Losses.MSE(), scalers=nothing,
            batchsize=8, dev=identity, rng, unsupported...)
    end
end

@testset "ModernNCA chunked equivalence" begin
    rng = Xoshiro(42)
    modernnca = NeuroTabModels.Models.ModernNCA
    Lux = modernnca.Lux

    for (loss, outsize, make_y) in (
        (NeuroTabModels.Losses.MSE(), 1, n -> randn(rng, Float32, n)),
        (NeuroTabModels.Losses.LogLoss(), 1,
            n -> Float32.(rand(rng, 0:1, n))),
        (NeuroTabModels.Losses.MLogLoss(), 5,
            n -> UInt32.(rand(rng, 1:5, n))),
    )
        model = NeuroTabModels.ModernNCAConfig(;
            d_embedding=16, n_blocks=0, corpus_chunk_size=777)(
            ; ins=16, outsize, loss)
        ps, st = Lux.setup(rng, model)
        st = Lux.testmode(st)
        y = make_y(5000)
        x = randn(rng, Float32, 16, 64)
        corpus = modernnca.Corpus(
            modernnca._stack(randn(rng, Float32, 16, 5000), 777),
            modernnca._stack(modernnca._target_layout(loss, y), 777), Dict(:nrounds => 0))

        chunked, _ = model((x, corpus), ps, st)
        zq, _ = modernnca._encode(model, x, ps, st)
        dense = dense_nca(model, zq, modernnca._keys(model, corpus, ps, st), y)
        @test chunked ≈ dense rtol=1f-4
    end

    info = Dict{Symbol,Any}(:nrounds => 0)
    mse_model = NeuroTabModels.ModernNCAConfig(; d_embedding=4, n_blocks=0)(
        ; ins=3, outsize=1, loss=NeuroTabModels.Losses.MSE())
    ps, st = Lux.setup(Xoshiro(7), mse_model)
    st = Lux.testmode(st)
    corpus = modernnca.Corpus(
        modernnca._stack(randn(rng, Float32, 3, 12), 5),
        modernnca._stack(randn(rng, Float32, 1, 12), 5), info)
    z1 = modernnca._keys(mse_model, corpus, ps, st)
    @test modernnca._keys(mse_model, corpus, ps, st) === z1
    info[:nrounds] = 1
    @test modernnca._keys(mse_model, corpus, ps, st) !== z1

    loader = modernnca.ModernNCALoader(
        randn(rng, Float32, 3, 20), randn(rng, Float32, 20),
        4, 10, 5, rng, identity, false)
    (_, cand_x, _, _), _ = first(loader)
    @test size(cand_x) == (3, 5, 2)
    @test size(unique(reshape(cand_x, 3, :); dims=2), 2) == 10

    x = randn(rng, Float32, 3, 2)
    moved_x, retained_corpus = Lux.cpu_device()((x, corpus))
    @test moved_x == x
    @test retained_corpus === corpus
end

@testset "ModernNCA reactant" begin
    rng = Xoshiro(123)
    x = randn(rng, Float32, 40, 4)
    df = DataFrame(x, :auto)
    df.y = x[:, 1] .- 0.5f0 .* x[:, 2]
    dtrain, deval = df[1:32, :], df[33:end, :]
    features = names(df, r"x")

    # n_blocks=1 so BatchNorm goes through tracing; chunk=7 leaves a corpus tail.
    arch = NeuroTabModels.ModernNCAConfig(;
        d_embedding=8, n_blocks=1, d_block=8, dropout=0.0, sample_rate=0.5, corpus_chunk_size=7)
    make(backend) = NeuroTabRegressor(arch;
        embedding_config=NeuroTabModels.LinearEmbeddings(; d_embedding=2),
        nrounds=3, early_stopping_rounds=3, batchsize=16,
        scale_target=false, backend, device=:cpu)
    fitkw = (; feature_names=features, target_name=:y, deval, verbosity=0)

    # Inference: Reactant matches the CPU path on a Zygote-trained model.
    model = NeuroTabModels.fit(make(:zygote), dtrain; fitkw...)
    p_cpu = model(deval; backend=:zygote, device=:cpu)
    p_rx = Array(model(deval; backend=:reactant, device=:cpu))
    @test isapprox(p_rx, p_cpu; rtol=1e-4)

    # Training gradient: Enzyme through the checkpointed traced loop matches Zygote through the rrule.
    m, ps, st = model.chain, model.info[:ps], Lux.trainmode(model.info[:st])
    xq = permutedims(Matrix{Float32}(dtrain[1:16, features]))
    cx = reshape(permutedims(Matrix{Float32}(dtrain[17:30, features])), 4, 7, 2)
    yq, cy = Float32.(dtrain.y[1:16]), reshape(Float32.(dtrain.y[17:30]), 7, 2)
    d = ((xq, cx, cy, yq), yq)
    loss = NeuroTabModels.Losses.MSE()
    g_zy = only(Zygote.gradient(p_ -> loss(m, p_, st, d)[1], ps))
    rdev = Lux.reactant_device()
    ts = Lux.Training.TrainState(m, rdev(ps), rdev(st), Optimisers.Adam(0.01f0))
    g_rx, _, _, _ = Lux.Training.compute_gradients(
        NeuroTabModels.Fit.get_ad_backend(Val(:reactant)), loss, rdev(d), ts)
    # Empty NamedTuple slots (e.g. unused embedding layers) are `nothing` for Zygote
    # and omitted for Enzyme; compare only array leaves.
    arrleaves(g) = Iterators.filter(x -> x isa AbstractArray, Lux.Functors.fleaves(g))
    zy_arr, rx_arr = collect(arrleaves(g_zy)), collect(arrleaves(g_rx))
    @test length(zy_arr) == length(rx_arr)
    for (a, b) in zip(zy_arr, rx_arr)
        @test isapprox(Array(b), a; rtol=1e-3, atol=1e-5)
    end

    # End to end under Reactant: metric moves and predictions are finite.
    model = NeuroTabModels.fit(make(:reactant), dtrain; fitkw...)
    metrics = model.info[:logger][:metrics][:metric]
    @test length(unique(metrics)) > 1
    @test all(isfinite, Array(model(deval; backend=:reactant, device=:cpu)))
end
