# Embeddings

Embeddings transform raw input columns before they reach a model backbone. Numerical
embeddings operate column-wise, a temporal embedding takes one named time column, and a
categorical embedding takes a named list of categorical columns. The three branches are
concatenated: numerical, temporal, then categorical.

With grouped training, `BatchNormEmbeddings` takes its statistics from the real
rows of each padded group only; the other embeddings map each observation
independently and carry the mask through. See [Grouped padding and masks](@ref).

## Available Embeddings

- `IdentityEmbedding`: leaves numerical features unchanged.
- `LinearEmbeddings`: projects each numerical feature to a learned vector.
- `PeriodicEmbeddings`: expands each feature with learned sinusoidal terms before projection.
- `PiecewiseLinearEmbeddings`: computes feature bins from the training data and embeds the resulting piecewise-linear encoding.
- `BatchNormEmbeddings`: batch-normalizes raw numerical features without expanding them.
- `LayerNormEmbeddings`: layer-normalizes raw numerical features without expanding them.
- `TemporalEmbeddings`: embeds one time column with Fourier features and an optional trend term.
- `CategoricalEmbeddings`: learns a vector per level of each listed categorical column; a level unseen at fit time gets its own vector.

## Configuration

Use `EmbeddingLayer` to combine a numerical embedding with optional temporal and
categorical branches. `temp` and `cat` name the columns they embed; `num` embeds every
other column in `feature_names`. A dictionary form is also available for model
configuration dictionaries.

```julia
EmbeddingLayer(PeriodicEmbeddings(; d_embedding=24))

EmbeddingLayer(;
    num=PiecewiseLinearEmbeddings(; bins=32),
    temp=TemporalEmbeddings(; feature=:date),
    cat=CategoricalEmbeddings(; features=[:store, :dept], d_embedding=8),
)

EmbeddingLayer(Dict(
    :embedding_type => :periodic,
    :d_embedding => 24,
    :temporal => Dict(:feature => :date),
    :categorical => Dict(:features => [:store, :dept]),
))
```

Piecewise-linear and temporal embeddings need training data when the embedding chain is
built. Use `needs_x_train` to check that requirement for a config.

## API

```@autodocs
Modules = [NeuroTabModels.Models.Embeddings]
Private = false
Order = [:type, :function]
```
