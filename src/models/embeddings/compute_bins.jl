"""
    compute_bins(X; n_bins=48)

Compute quantile-based bin boundaries for `PiecewiseLinearEncoding`/`PiecewiseLinearEmbeddings`.

Note: `X` should have shape `(n_samples, n_features)`, which is the transpose of the
model input convention `(n_features, batch)`. Transpose your data before calling this.

# Arguments
- `X::AbstractMatrix`: Training data of shape `(n_samples, n_features)`.
- `n_bins::Int`: Number of bins per feature (default `48`).

# Returns
- `Vector{Vector{Float32}}`: Bin edges for each feature. Each vector has between 2 and
  `n_bins + 1` elements (fewer if quantiles coincide for low-cardinality features).
"""
function compute_bins(X::AbstractMatrix; n_bins::Int=48)
    n_samples, n_features = size(X)
    @assert n_bins > 1 "n_bins must be > 1, got $n_bins"
    @assert n_bins < n_samples "n_bins must be < n_samples, got n_bins=$n_bins, n_samples=$n_samples"

    quantile_probs = range(0f0, 1f0, length=n_bins + 1)

    bins = Vector{Vector{Float32}}(undef, n_features)
    for j in 1:n_features
        col = sort(X[:, j])
        edges = [Float32(quantile(col, p)) for p in quantile_probs]
        bins[j] = unique(edges)
        @assert length(bins[j]) >= 2 "Feature $j has fewer than 2 unique bin edges"
    end

    return bins
end