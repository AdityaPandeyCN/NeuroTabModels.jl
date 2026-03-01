# test_embeddings.jl
# 1. python generate_embedding_reference.py
# 2. julia --project=. test_embeddings.jl

using NeuroTabModels, Lux, Random, NPZ
using NeuroTabModels.Models.Embeddings: LinearEmbeddings, LinearReLUEmbeddings,
    PeriodicEmbeddings, Periodic, NLinear, PiecewiseLinearEmbeddings,
    PiecewiseLinearEncoding, compute_bins

const REF = NPZ.npzread("embedding_reference_data.npz")
const ATOL = 1e-5
const B, N_FEAT, D_EMB = 8, 5, 12

rng = Random.MersenneTwister(0)

# Python (B, n_feat, d_emb) → Julia (d_emb, n_feat, B)
t3(x) = permutedims(x, (3, 2, 1))
# Python (n_feat, d_emb) → Julia (d_emb, n_feat)
t2(x) = permutedims(x, (2, 1))
# Python (n, in, out) → Julia (out, in, n)
tw(x) = permutedims(x, (3, 2, 1))
# Python (B, n_feat) → Julia (n_feat, B)
tx(x) = permutedims(x, (2, 1))

function check(name, jl, py; atol=ATOL)
    diff = maximum(abs.(jl .- py))
    @assert diff < atol "$name: max diff $diff ≥ $atol"
    println("✅ $name (maxdiff=$(round(diff; sigdigits=2)))")
end

x = tx(REF["x"])

# ── LinearEmbeddings ─────────────────────────────────────────────

local y, _st

le = LinearEmbeddings(N_FEAT, D_EMB)
ps_le = (weight=t2(REF["linear_weight"]), bias=t2(REF["linear_bias"]))
y, _st = le(x, ps_le, (;))
check("LinearEmbeddings", y, t3(REF["y_linear"]))

# ── LinearReLUEmbeddings ─────────────────────────────────────────

lre = LinearReLUEmbeddings(N_FEAT, D_EMB)
ps_lre = (layer=(weight=t2(REF["linear_relu_weight"]), bias=t2(REF["linear_relu_bias"])),)
y, _st = lre(x, ps_lre, (layer=(;),))
check("LinearReLUEmbeddings", y, t3(REF["y_linear_relu"]))

# ── PeriodicEmbeddings (lite=false) ──────────────────────────────

pe = PeriodicEmbeddings(N_FEAT, D_EMB; n_frequencies=16, frequency_init_scale=0.01f0, lite=false)
ps_pe = (
    periodic=(weight=t2(REF["periodic_freq"]),),
    linear=(weight=tw(REF["periodic_linear_weight"]),
            bias=reshape(t2(REF["periodic_linear_bias"]), D_EMB, 1, N_FEAT)),
)
st_pe = Lux.initialstates(rng, pe)
y, _st = pe(x, ps_pe, st_pe)
check("PeriodicEmbeddings", y, t3(REF["y_periodic"]))

# ── PeriodicEmbeddings (lite=true) ───────────────────────────────

pe_lite = PeriodicEmbeddings(N_FEAT, D_EMB; n_frequencies=16, frequency_init_scale=0.01f0, lite=true)
ps_pe_lite = (
    periodic=(weight=t2(REF["periodic_lite_freq"]),),
    linear=(weight=REF["periodic_lite_linear_weight"],
            bias=REF["periodic_lite_linear_bias"]),
)
st_pe_lite = Lux.initialstates(rng, pe_lite)
y, _st = pe_lite(x, ps_pe_lite, st_pe_lite)
check("PeriodicEmbeddings (lite)", y, t3(REF["y_periodic_lite"]))

# ── PiecewiseLinearEmbeddings (version B) ─────────────────────────

n_bf = Int(REF["n_bin_features"])
bins = [Vector{Float32}(REF["bin_$i"]) for i in 0:n_bf-1]

ple_b = PiecewiseLinearEmbeddings(bins, D_EMB; activation=false, version=:B)
st_ple_b = Lux.initialstates(rng, ple_b)

ps_ple_b = (
    linear0=(weight=t2(REF["ple_b_linear0_weight"]),
             bias=t2(REF["ple_b_linear0_bias"])),
    encoding=(;),
    linear=(weight=tw(REF["ple_b_linear_weight"]),),
)
y, _st = ple_b(x, ps_ple_b, st_ple_b)
check("PiecewiseLinearEmbeddings (B)", y, t3(REF["y_ple_b"]))

# ── PiecewiseLinearEmbeddings (version A) ─────────────────────────

ple_a = PiecewiseLinearEmbeddings(bins, D_EMB; activation=false, version=:A)
st_ple_a = Lux.initialstates(rng, ple_a)

ps_ple_a = (
    linear0=nothing,
    encoding=(;),
    linear=(weight=tw(REF["ple_a_linear_weight"]),
            bias=reshape(t2(REF["ple_a_linear_bias"]), D_EMB, 1, N_FEAT)),
)
y, _st = ple_a(x, ps_ple_a, st_ple_a)
check("PiecewiseLinearEmbeddings (A)", y, t3(REF["y_ple_a"]))

println("\n🎉 All embeddings match rtdl_num_embeddings v0.0.12 (atol=$ATOL)")