# test_tabm.jl — julia --project=. test_tabm.jl

using NeuroTabModels, Lux, Random, Enzyme
using NeuroTabModels.Models.Embeddings: compute_bins

rng = Random.MersenneTwister(42)
nfeats, batch, outsize = 10, 32, 1
x = randn(Float32, nfeats, batch)

function test_forward(name, config)
    chain = config(; nfeats, outsize)
    ps, st = Lux.setup(rng, chain)
    y, _ = chain(x, ps, st)
    @assert size(y) == (outsize, batch) "$name: expected ($outsize,$batch), got $(size(y))"
    println("✅ $name: $(size(y))")
    return chain, ps, st
end

# Forward passes
test_forward("TabM",        TabMConfig(; k=4, n_blocks=2, d_block=64, dropout=0.0))
test_forward("TabM-mini",   TabMConfig(; k=4, n_blocks=2, d_block=64, dropout=0.0, arch_type=:tabm_mini))
test_forward("TabM-packed", TabMConfig(; k=4, n_blocks=2, d_block=64, dropout=0.0, arch_type=:tabm_packed))
test_forward("Periodic",    TabMConfig(; k=4, n_blocks=2, d_block=64, dropout=0.0, use_embeddings=true, embedding_type=:periodic))
test_forward("Linear",      TabMConfig(; k=4, n_blocks=2, d_block=64, dropout=0.0, use_embeddings=true, embedding_type=:linear))

bins = compute_bins(randn(Float32, 200, nfeats); n_bins=48)
chain, ps, st = test_forward("PLE", TabMConfig(; k=4, n_blocks=2, d_block=64, dropout=0.0, use_embeddings=true, embedding_type=:piecewise, bins=bins))

# Gradient (Enzyme)
dps = Enzyme.make_zero(ps)
Enzyme.autodiff(Reverse, (ps, st, x) -> sum(first(chain(x, ps, st))),
    Active, Duplicated(ps, dps), Const(st), Const(x))
@assert any(v -> v != 0, Lux.fmap(x -> x isa AbstractArray ? sum(abs, x) : 0f0, dps))
println("✅ Enzyme gradients")

println("\n🎉 All passed")