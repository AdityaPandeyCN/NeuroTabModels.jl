using NeuroTabModels, Lux, Random, NPZ, Statistics
using NeuroTabModels.Models.TabM: LinearBatchEnsemble, LinearEnsemble, TabMConfig
using NeuroTabModels.Models.Embeddings: LinearEmbeddings, PeriodicEmbeddings, PiecewiseLinearEmbeddings, PiecewiseLinearEncoding, NLinear, Periodic, compute_bins
using NeuroTabModels.Losses: reduce_pred

const REF = NPZ.npzread("tabm_reference_data.npz")
const ATOL = 1e-5
const B, K, D_IN, D_OUT, D_BLOCK, N_BLOCKS = 8, 4, 10, 1, 32, 2
const D_EMB, N_FREQ, N_BINS = 16, 24, 8

rng = Random.MersenneTwister(0)

t2(a) = permutedims(a, (2, 1))
t3(a) = permutedims(a, (3, 2, 1))

x   = t2(REF["x"])
x3d = t3(REF["x3d"])

function check(name, y, y_ref)
    md = maximum(abs.(y .- y_ref))
    pass = md < ATOL
    println("  $(pass ? "✓" : "✗") $name  maxdiff=$md  shape=$(size(y))")
    @assert pass "$name failed: maxdiff=$md >= $ATOL"
end

# LinearBatchEnsemble 
# PyTorch: weight (32,10), r (4,10), s (4,32), bias (4,32)
# Julia:   weight (32,10), r (10,4), s (32,4), bias (32,4)

println("LinearBatchEnsemble")
ps_lbe = (
    weight = REF["lbe_weight"],       # (32, 10) — same convention
    r      = t2(REF["lbe_r"]),        # (4, 10) → (10, 4)
    s      = t2(REF["lbe_s"]),        # (4, 32) → (32, 4)
    bias   = t2(REF["lbe_bias"]),     # (4, 32) → (32, 4)
)
y_lbe, _ = LinearBatchEnsemble(D_IN, D_BLOCK; k=K, scaling_init=:random_signs)(x3d, ps_lbe, (;))
check("forward", y_lbe, t3(REF["y_lbe"]))

# LinearEnsemble 

println("LinearEnsemble")
le_in = t3(REF["le_input"])
ps_le = (weight = t3(REF["le_weight"]), bias = t2(REF["le_bias"]))
y_le, _ = LinearEnsemble(D_BLOCK, D_OUT, K)(le_in, ps_le, (;))
check("forward", y_le, t3(REF["y_le"]))

# Full model smoke tests

println("Full models (smoke)")
for arch in [:tabm, :tabm_mini, :tabm_packed]
    cfg   = TabMConfig(; k=K, n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=0.0, arch_type=arch)
    chain = cfg(; nfeats=D_IN, outsize=D_OUT)
    ps, st = Lux.setup(rng, chain)
    y, _ = chain(x, ps, Lux.testmode(st))
    y_avg = reduce_pred(y)
    @assert size(y) == (D_OUT, K, B) "$arch raw shape"
    @assert size(y_avg) == (D_OUT, B) "$arch reduced shape"
    @assert !any(isnan, y) "$arch NaN"
    println("  ✓ $arch  raw=$(size(y))  reduced=$(size(y_avg))")
end

# LinearEmbeddings 
# PyTorch: weight (10, 16), bias (10, 16), output (8, 10, 16)
# Julia:   weight (16, 10, 1), bias (16, 10, 1), output (16, 10, 8)

println("LinearEmbeddings")
x_emb = t2(REF["x_emb"])

ps_lin_emb = (
    weight = reshape(t2(REF["lin_emb_weight"]), D_EMB, D_IN, 1),
    bias   = reshape(t2(REF["lin_emb_bias"]),   D_EMB, D_IN, 1),
)
y_lin_emb, _ = LinearEmbeddings(D_IN, D_EMB)(x_emb, ps_lin_emb, (;))
check("forward", y_lin_emb, t3(REF["y_lin_emb"]))

# PeriodicEmbeddings 
# PyTorch params:
#   periodic.weight (10, 24)        — raw, 2π applied in forward
#   linear.weight   (10, 48, 16)    — NLinear weight
#   linear.bias     (10, 16)        — NLinear bias
# Julia params:
#   periodic.weight (24, 10, 1)     — 2π baked in
#   linear.weight   (16, 48, 10)    — (d_emb, 2*n_freq, n_features)
#   linear.bias     (16, 1, 10)     — (d_emb, 1, n_features)

println("PeriodicEmbeddings")
per_emb = PeriodicEmbeddings(D_IN, D_EMB; n_frequencies=N_FREQ,
                              frequency_init_scale=0.01f0, activation=true, lite=false)

per_w = reshape(Float32(2π) .* t2(Float32.(REF["per_periodic_weight"])), N_FREQ, D_IN, 1)
nlin_w = permutedims(Float32.(REF["per_linear_weight"]), (3, 2, 1))
nlin_b = reshape(t2(Float32.(REF["per_linear_bias"])), D_EMB, 1, D_IN)

ps_per_emb = (
    periodic = (weight = per_w,),
    linear   = (weight = nlin_w, bias = nlin_b),
)
st_per_emb = Lux.initialstates(rng, per_emb)

y_per_emb, _ = per_emb(x_emb, ps_per_emb, st_per_emb)
check("forward", y_per_emb, t3(REF["y_per_emb"]))

# PiecewiseLinearEmbeddings (version B) 
# PyTorch params:
#   linear0.weight (10, 16)     — LinearEmbeddings weight
#   linear0.bias   (10, 16)     — LinearEmbeddings bias
#   linear.weight  (10, 8, 16)  — NLinear weight (zero-init for version B)
# Julia params:
#   linear0.weight (16, 10, 1)
#   linear0.bias   (16, 10, 1)
#   linear.weight  (16, 8, 10)  — (d_emb, max_n_bins, n_features)
# NOTE: Isolated PiecewiseLinearEncoding skipped because PyTorch version
# outputs flattened 2D (B, total_bins), Julia outputs 3D (max_bins, n_features, B).
# The full PLE test covers encoding correctness implicitly.

println("PiecewiseLinearEmbeddings")
bins = Vector{Vector{Float32}}(undef, D_IN)
for i in 0:D_IN-1
    bins[i+1] = Float32.(REF["ple_bins_$i"])
end

ple = PiecewiseLinearEmbeddings(bins, D_EMB; activation=false, version=:B)

l0_w = reshape(t2(Float32.(REF["ple_linear0_weight"])), D_EMB, D_IN, 1)
l0_b = reshape(t2(Float32.(REF["ple_linear0_bias"])),   D_EMB, D_IN, 1)
nlin_w_ple = permutedims(Float32.(REF["ple_linear_weight"]), (3, 2, 1))

ps_ple = (
    linear0  = (weight = l0_w, bias = l0_b),
    encoding = (;),
    linear   = (weight = nlin_w_ple,),
)
st_ple = Lux.initialstates(rng, ple)

y_ple, _ = ple(x_emb, ps_ple, st_ple)
check("forward", y_ple, t3(REF["y_ple"]))

# TabM† test

println("TabM† ")
X_train_big = randn(Float32, 64, D_IN)
cfg_dag = TabMConfig(; k=K, n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=0.0,
                       arch_type=:tabm_mini, use_embeddings=true, embedding_type=:piecewise,
                       d_embedding=D_EMB, n_bins=N_BINS)
chain_dag = cfg_dag(; nfeats=D_IN, outsize=D_OUT, X_train=X_train_big)
ps_dag, st_dag = Lux.setup(rng, chain_dag)
y_dag, _ = chain_dag(x_emb, ps_dag, Lux.testmode(st_dag))
y_dag_avg = reduce_pred(y_dag)

@assert size(y_dag) == (D_OUT, K, B) "TabM† raw shape"
@assert size(y_dag_avg) == (D_OUT, B) "TabM† reduced shape"
@assert !any(isnan, y_dag) "TabM† NaN"
println("  ✓ tabm_mini†  raw=$(size(y_dag))  reduced=$(size(y_dag_avg))")

println("\n✓ all passed")