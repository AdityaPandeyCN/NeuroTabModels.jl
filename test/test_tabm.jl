# test_tabm.jl
# 1. python generate_reference.py
# 2. julia --project=. test_tabm.jl

using NeuroTabModels, Lux, Random, NPZ
using NeuroTabModels.Models.TabM: LinearBatchEnsemble, LinearEnsemble, SharedDense,
    ScaleEnsemble, EnsembleView, MeanEnsemble

const REF = NPZ.npzread("tabm_reference_data.npz")
const ATOL = 1e-5
const B, K, D_IN, D_OUT, D_BLOCK, N_BLOCKS = 8, 4, 10, 1, 32, 2
rng = Random.MersenneTwister(0)

t2(x) = permutedims(x, (2, 1))                      # (K,D) → (D,K)
t3(x) = permutedims(x, (3, 2, 1))                   # (B,K,D) → (D,K,B)
avgK(py) = dropdims(sum(t3(py); dims=2); dims=2) ./ K  # Python (B,K,D) → Julia (D,B)

function check(name, jl, py; atol=ATOL)
    diff = maximum(abs.(jl .- py))
    @assert diff < atol "$name: max diff $diff ≥ $atol"
    println("✅ $name (maxdiff=$(round(diff; sigdigits=2)))")
end

x, x3d = t2(REF["x"]), t3(REF["x3d"])

# ── Layer: LinearBatchEnsemble ───────────────────────────────────

y, _ = LinearBatchEnsemble(D_IN, D_BLOCK; k=K, scaling_init=:random_signs)(
    x3d, (weight=REF["lbe_weight"], r=t2(REF["lbe_r"]),
           s=t2(REF["lbe_s"]), bias=t2(REF["lbe_bias"])), (;))
check("LinearBatchEnsemble", y, t3(REF["y_lbe"]))

# ── Layer: LinearEnsemble ────────────────────────────────────────

y, _ = LinearEnsemble(D_BLOCK, D_OUT, K)(
    t3(REF["le_input"]),
    (weight=t3(REF["le_weight"]), bias=t2(REF["le_bias"])), (;))
check("LinearEnsemble", y, t3(REF["y_le"]))

# ── Full TabM ────────────────────────────────────────────────────

function load_tabm_ps(pfx)
    d = Dict{Symbol,Any}(:layer_1 => (;))
    for blk in 0:N_BLOCKS-1
        p = "$(pfx)backbone_blocks_$(blk)_0_"
        d[Symbol("layer_$(2+blk*2)")] = (
            weight=REF["$(p)weight"], r=t2(REF["$(p)r"]),
            s=t2(REF["$(p)s"]), bias=t2(REF["$(p)bias"]))
        d[Symbol("layer_$(3+blk*2)")] = (;)
    end
    d[Symbol("layer_$(2+N_BLOCKS*2)")] = (
        layer_1=(weight=t3(REF["$(pfx)output_weight"]),
                 bias=t2(REF["$(pfx)output_bias"])),
        layer_2=(;))
    NamedTuple(d)
end

function load_mini_ps(pfx)
    d = Dict{Symbol,Any}(:layer_1 => (;),
        :layer_2 => (weight=t2(REF["$(pfx)backbone_affine_weight"]),))
    for blk in 0:N_BLOCKS-1
        p = "$(pfx)backbone_blocks_$(blk)_0_"
        d[Symbol("layer_$(3+blk*2)")] = (weight=REF["$(p)weight"], bias=REF["$(p)bias"])
        d[Symbol("layer_$(4+blk*2)")] = (;)
    end
    d[Symbol("layer_$(3+N_BLOCKS*2)")] = (
        layer_1=(weight=t3(REF["$(pfx)output_weight"]),
                 bias=t2(REF["$(pfx)output_bias"])),
        layer_2=(;))
    NamedTuple(d)
end

function load_packed_ps(pfx)
    d = Dict{Symbol,Any}(:layer_1 => (;))
    for blk in 0:N_BLOCKS-1
        p = "$(pfx)backbone_blocks_$(blk)_0_"
        d[Symbol("layer_$(2+blk*2)")] = (weight=t3(REF["$(p)weight"]), bias=t2(REF["$(p)bias"]))
        d[Symbol("layer_$(3+blk*2)")] = (;)
    end
    d[Symbol("layer_$(2+N_BLOCKS*2)")] = (
        layer_1=(weight=t3(REF["$(pfx)output_weight"]),
                 bias=t2(REF["$(pfx)output_bias"])),
        layer_2=(;))
    NamedTuple(d)
end

for (name, arch, loader, pfx) in [
    ("TabM",        :tabm,        load_tabm_ps,   "tabm_"),
    ("TabM-mini",   :tabm_mini,   load_mini_ps,   "mini_"),
    ("TabM-packed", :tabm_packed, load_packed_ps,  "packed_"),
]
    cfg = TabMConfig(; k=K, n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=0.0, arch_type=arch)
    chain = cfg(; nfeats=D_IN, outsize=D_OUT)
    st = Lux.testmode(Lux.initialstates(rng, chain))
    y, _ = chain(x, loader(pfx), st)
    check("$name full", y, avgK(REF["y_$(pfx[1:end-1])"]))
end

println("\n🎉 All match official tabm (atol=$ATOL)")