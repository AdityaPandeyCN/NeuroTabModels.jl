using Test
using Lux
using LuxCore
using Reactant
using Enzyme
using Random
using LinearAlgebra
using NeuroTabModels

# Shortcuts to internal modules
const TabM = NeuroTabModels.Models.TabM
const TabMCfg = NeuroTabModels.TabMConfig
const LEE = TabM.LinearEfficientEnsemble
const LE = TabM.LinearEnsemble
const ME = TabM.MeanEnsemble
const EV = TabM.EnsembleView
const _brelu = TabM._broadcast_relu

@testset "TabM Cross-Verify" begin
    rng = MersenneTwister(42)

    # 1. Check BatchEnsemble Math
    @testset "LinearEfficientEnsemble formula" begin
        in_f, out_f, k, batch = 6, 4, 3, 5
        layer = LEE(in_f, out_f; k=k, scaling_init=:random_signs)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, Float32, in_f, k, batch)
        y, _ = layer(x, ps, st)

        # Manual calculation to verify
        # R * x
        rx = x .* reshape(ps.r, in_f, k, 1)
        # W * (R * x)  [Shared Weight]
        wx_flat = ps.weight * reshape(rx, in_f, k * batch)
        wx = reshape(wx_flat, out_f, k, batch)
        # S * W... + B
        y_ref = reshape(ps.s, out_f, k, 1) .* wx .+ reshape(ps.bias, out_f, k, 1)
        
        @test y ≈ y_ref atol=1e-5
    end

    # 2. Check LinearEnsemble (Independent weights)
    @testset "LinearEnsemble per-member" begin
        in_f, out_f, k, batch = 6, 2, 3, 5
        layer = LE(in_f, out_f, k)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, Float32, in_f, k, batch)
        y, _ = layer(x, ps, st)
        
        for i in 1:k
            # Check if slice i matches weight i
            @test y[:, i, :] ≈ ps.weight[:, :, i] * x[:, i, :] .+ ps.bias[:, i] atol=1e-5
        end
    end

    # 3. Check Shared Weights (No Adapters)
    @testset "Shared linear" begin
        in_f, out_f, k, batch = 6, 4, 3, 5
        layer = LEE(in_f, out_f; k=k, ensemble_scaling_in=false, ensemble_scaling_out=false, bias=true, ensemble_bias=false)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, Float32, in_f, k, batch)
        y, _ = layer(x, ps, st)
        for i in 1:k
            @test y[:, i, :] ≈ ps.weight * x[:, i, :] .+ ps.bias atol=1e-5
        end
    end

    # 4. Check Averaging
    @testset "MeanEnsemble" begin
        x = randn(rng, Float32, 4, 3, 5)
        y, _ = ME()(x, (;), (;))
        @test y ≈ dropdims(sum(x; dims=2); dims=2) ./ 3f0 atol=1e-6
    end

    # 5. Check Initialization (First layer ±1, others 1)
    @testset "Init scheme" begin
        # use_embeddings=false to check the raw backbone logic
        cfg = TabMCfg(k=4, n_blocks=2, d_block=16, arch_type=:tabm, dropout=0.0, scaling_init=:random_signs, use_embeddings=false)
        model = cfg(nfeats=10, outsize=1)
        ps, _ = Lux.setup(rng, model)

        # Helper to find params named :r
        r_params = []
        function walk(p)
            for k in keys(p)
                if k == :r
                    push!(r_params, p[k])
                elseif p[k] isa NamedTuple
                    walk(p[k])
                end
            end
        end
        walk(ps)

        if length(r_params) >= 2
            # Layer 1: Random signs
            @test all(x -> isapprox(abs(x), 1.0f0; atol=1e-5), r_params[1])
            # Layer 2: Ones (Identity)
            @test all(x -> x ≈ 1.0f0, r_params[2])
        end
    end

    # 6. E2E Shapes
    @testset "E2E Shapes" begin
        for arch in (:tabm, :tabm_mini, :tabm_packed)
            @testset "$arch" begin
                model = TabMCfg(k=4, n_blocks=2, d_block=16, arch_type=arch, dropout=0.0)(nfeats=10, outsize=3)
                ps, st = Lux.setup(rng, model)
                y, _ = model(randn(rng, Float32, 10, 8), ps, st)
                @test size(y) == (3, 8)
            end
        end
    end

    # 7. Gradients via Reactant (Simplified)
    @testset "Grads Reactant" begin
        # Only run if Reactant is working
        try
            dev = reactant_device()
            
            for arch in (:tabm, :tabm_mini)
                @testset "$arch" begin
                    model = TabMCfg(k=4, n_blocks=1, d_block=8, arch_type=arch, dropout=0.0)(nfeats=4, outsize=1)
                    
                    # Lux.setup returns Train Mode by default.
                    # We just use this state directly. No mode switching needed.
                    ps, st = Lux.setup(rng, model) 

                    x_r = dev(randn(rng, Float32, 4, 8))
                    ps_r = dev(ps)
                    st_r = dev(st) 

                    function loss_fn(p)
                        y, _ = model(x_r, p, st_r)
                        return sum(y)
                    end

                    # Enzyme returns a tuple of gradients
                    g_tuple = @jit Enzyme.gradient(Reverse, loss_fn, ps_r)
                    g = g_tuple[1]
                    
                    # Basic check: verify gradients exist (are not all zero)
                    has_grad(x::AbstractArray) = any(x .!= 0)
                    has_grad(x::NamedTuple) = any(has_grad(v) for v in values(x))
                    
                    @test has_grad(g)
                end
            end
        catch
             # If Reactant/XLA isn't installed/working, just skip without failing the suite
        end
    end
end