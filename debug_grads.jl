using Pkg
Pkg.activate(".")

using Flux
using Enzyme
using Statistics

println("=== Direct Enzyme Test ===\n")

# Simple model
model = Chain(Dense(4 => 8, relu), Dense(8 => 3))

# Test data
x = randn(Float32, 4, 16)
y = UInt32.(rand(1:3, 16))

# Loss function 
function test_mlogloss(m, x, y)
    logits = m(x)
    p = Flux.logsoftmax(logits; dims=1)
    k, n = size(p)
    class_range = reshape(1:k, k, 1)
    labels = reshape(Int.(y), 1, n)
    y_onehot = eltype(p).(class_range .== labels)
    return -sum(y_onehot .* p) / n
end

loss_val = test_mlogloss(model, x, y)
println("Loss value: ", loss_val)

# Zygote
println("\n--- Zygote ---")
grads_zygote = Flux.gradient(m -> test_mlogloss(m, x, y), model)[1]
println("Zygote works! First layer weight grad norm: ", sum(abs, grads_zygote.layers[1].weight))

# Now test Enzyme directly (not via Flux.gradient)
println("\n--- Enzyme Direct ---")
Enzyme.API.strictAliasing!(false)

# Create shadow manually
dmodel = Flux.fmap(model) do x
    x isa AbstractArray ? zero(x) : x
end

println("Shadow created")

# Try autodiff directly
try
    Enzyme.autodiff(
        Enzyme.Reverse,
        m -> test_mlogloss(m, x, y),
        Enzyme.Active,
        Enzyme.Duplicated(model, dmodel)
    )
    println("Enzyme autodiff completed!")
    println("Enzyme first layer weight grad norm: ", sum(abs, dmodel.layers[1].weight))
catch e
    println("Enzyme error: ", e)
end

# Compare
println("\n--- Comparison ---")
println("Zygote norm: ", sum(abs, grads_zygote.layers[1].weight))
println("Enzyme norm: ", sum(abs, dmodel.layers[1].weight))