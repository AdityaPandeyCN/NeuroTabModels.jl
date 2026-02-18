module Embeddings

using Lux
using Random
using NNlib
import Statistics: quantile

export NLinear, LinearEmbeddings, LinearReLUEmbeddings
export Periodic, PeriodicEmbeddings
export PiecewiseLinearEncoding, PiecewiseLinearEmbeddings
export compute_bins

# Utilities
include("compute_bins.jl")
include("nlinear.jl")

# Embedding Layers
include("linear.jl")
include("periodic.jl")
include("piecewise_linear.jl")

end