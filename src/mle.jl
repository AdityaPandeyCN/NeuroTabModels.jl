using NNlib: softplus
using Statistics: mean
using Reactant
using Enzyme

p = Float32[0.0 0.0 0.0;
            0.0 0.0 0.0]
y = Float32[1.0, -1.0, 0.5]

# ── Original formulation components ──

# Test 1: exp(2σ)
function loss_exp2sigma(p, y)
    σ = view(p, 2, :)
    mean(exp.(2 .* σ))
end

# Test 2: max(eps, exp(2σ))
function loss_max_exp(p, y)
    σ = view(p, 2, :)
    T = eltype(p)
    mean(max.(T(2e-7), exp.(2 .* σ)))
end

# Test 3: (y - μ)² / (2 * max(eps, exp(2σ)))
function loss_ratio(p, y)
    μ = view(p, 1, :)
    σ = view(p, 2, :)
    T = eltype(p)
    mean((y .- μ) .^ 2 ./ (2 .* max.(T(2e-7), exp.(2 .* σ))))
end

# Test 4: -σ (just the log-det term)
function loss_neg_sigma(p, y)
    σ = view(p, 2, :)
    mean(-σ)
end

# Test 5: -σ - (y-μ)²/(2*exp(2σ)) — the log-likelihood
function loss_loglik(p, y)
    μ = view(p, 1, :)
    σ = view(p, 2, :)
    T = eltype(p)
    mean(-σ .- (y .- μ) .^ 2 ./ (2 .* max.(T(2e-7), exp.(2 .* σ))))
end

# Test 6: negated log-likelihood — the actual original loss
function loss_original(p, y)
    μ = view(p, 1, :)
    σ = view(p, 2, :)
    T = eltype(p)
    mean(-(-σ .- (y .- μ) .^ 2 ./ (2 .* max.(T(2e-7), exp.(2 .* σ)))))
end

# Test 7: manual softplus full MLE (our fix)
_softplus(x) = log(one(x) + exp(x))
function loss_fixed(p, y)
    μ = view(p, 1, :)
    raw_σ = view(p, 2, :)
    T = eltype(p)
    σ = _softplus.(raw_σ) .+ T(1e-4)
    mean(log.(σ) .+ (y .- μ) .^ 2 ./ (2 .* σ .^ 2))
end

# ── Gradient helpers ──

function cpu_grad(loss_fn, p, y)
    dp = zero(p)
    Enzyme.autodiff(Reverse, loss_fn, Active, Duplicated(p, dp), Const(y))
    return dp
end

function reactant_grad(loss_fn, p, y)
    function _grad(p, y)
        dp = zero(p)
        Enzyme.autodiff(Reverse, loss_fn, Active, Duplicated(p, dp), Const(y))
        return dp
    end
    local p_r = Reactant.to_rarray(copy(p))
    local y_r = Reactant.to_rarray(copy(y))
    compiled = @compile _grad(p_r, y_r)
    return Array(compiled(Reactant.to_rarray(copy(p)), Reactant.to_rarray(copy(y))))
end

# ── Run tests ──

tests = [
    ("1: exp(2σ)",                    loss_exp2sigma),
    ("2: max(eps, exp(2σ))",          loss_max_exp),
    ("3: (y-μ)²/(2*max(eps,exp(2σ)))", loss_ratio),
    ("4: -σ (neg sigma)",             loss_neg_sigma),
    ("5: log-likelihood",             loss_loglik),
    ("6: ORIGINAL loss (neg loglik)", loss_original),
    ("7: FIXED (manual softplus)",    loss_fixed),
]

for (name, fn) in tests
    println("="^60)
    println("TEST $name")
    println("="^60)

    g_cpu = cpu_grad(fn, copy(p), y)
    g_xla = reactant_grad(fn, p, y)

    println("  CPU μ grads: $(g_cpu[1, :])")
    println("  XLA μ grads: $(g_xla[1, :])")
    println("  CPU σ grads: $(g_cpu[2, :])")
    println("  XLA σ grads: $(g_xla[2, :])")

    sign_match_1 = all(sign.(g_cpu[1, :]) .== sign.(g_xla[1, :]))
    sign_match_2 = all(sign.(g_cpu[2, :]) .== sign.(g_xla[2, :]))
    values_match = isapprox(g_cpu, g_xla; atol=1e-5)

    if values_match
        println("  ✓ PERFECT MATCH")
    else
        match_str(x) = x ? "✓ OK" : "✗ SIGN FLIP"
        println("  Row 1 (μ): $(match_str(sign_match_1))")
        println("  Row 2 (σ): $(match_str(sign_match_2))")
    end
    println()
end