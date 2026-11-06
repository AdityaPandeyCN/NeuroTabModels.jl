# Grouped padding and masks

`MLPAttn` and `NeuroTreeAttn` let information travel across observations that
belong to the same group. A grouped loader (`group_name`) uses `batchsize=0`:
each step is one group, padded to a common width. Attention’s own batch
dimension is therefore always `1` — the sequence is the group.

This page covers that padding: where `w` comes from, which layers read it,
and how the rest of the chain carries it.

## Why there is a pad at all

A grouped train/eval/infer loader takes `groupby(df, group_name)` and writes
**one padded tensor per group**. The buffer width is the size of the largest
group, so every group is a dense `(features, buffer)` matrix:

```
one group, 3 real rows, buffer = 5

        col:   1    2    3    4    5
               ├──── real ────┤  pads

   x  =  [  x₁   x₂   x₃   0    0  ]     (nfeats × buffer)
   y  =  [  y₁   y₂   y₃   0    0  ]
   w  =  [  w₁   w₂   w₃   0    0  ]     row weight (1 without weight_name), 0 = pad slot
   o  =  [  o₁   o₂   o₃   0    0  ]     offset, with offset_name
```

Train/eval use `w` as the row weight, 0 on pads (shape `(1, 1, buffer)`). Infer uses a
boolean `mask` of length `buffer` and, after the forward, keeps only
`pred[:, mask]`.

Pads exist so the group is a rectangular array. They are not observations.
Anything that reduces across the last dimension must ignore them. Per-column
maps (embeddings, Dense heads) can run on pad columns without contaminating
real ones.

## Two jobs for `w`

`w` is reused for two different things, at two different places:

| Job | Who uses it | What happens to pads |
| --- | --- | --- |
| **Loss / metric weight** | `_reduce(loss, w)` and eval | pad slots contribute `0` to the scalar loss |
| **Padding mask** | `MaskedBatchNorm`, attention | pads are excluded from BatchNorm stats and from attention keys |

Any layer that reduces or mixes across the group needs the second job. A pad
token with value `0` would still be a key every query can attend to, and
BatchNorm is a reduction over the group: pad zeros would enter `μ` / `σ` and
running stats. Those layers read `w > 0` as a valid-token flag; sample weights
never change BatchNorm statistics.

## Who reads `w`

Every layer takes either `x` or `(x, w)` and returns the same form. Layers that
reduce or mix across the group read `w`; per-observation layers carry it:

| Layer | With `(x, w)` |
| --- | --- |
| `MaskedBatchNorm` (MLP, ResNet, `*Attn` encoders, `BatchNormEmbeddings`) | μ, σ and running stats from valid tokens only |
| `AttnResidual` | padded keys masked for every query |
| `CarryMask(layer)` (Dense, Dropout, per-observation embeddings) | `(layer(x), w)` |
| `MaskSkip(layer)` | `(x + layer(x), w)` |

Embedding branches are routed with a `BranchLayer`, which hands `(x, w)` whole
to each branch, and their outputs are concatenated with `w` carried once.

```
input is either x  or  (x, w)

                    (x, w)
                       │
          ┌────────────┴────────────┐
          │      MaskedModel        │
          │                         │
          │   (z, w) = embed(x, w)  │  ← BatchNormEmbeddings reads w
          │                         │
          │   y = core((z, w))      │  ← MaskedBN / attention read w
          └────────────┬────────────┘
                       │
                       y
```

`fit` chooses the wrapper once:

```julia
chain = if uses_batch_mask(core_chain)
    MaskedModel(embed_chain, core_chain)   # MLP, ResNet, MLPAttn, NeuroTreeAttn
elseif !isnothing(group_name)
    # the core takes `x` alone, but the embedding may still hold batch statistics
    MaskedModel(embed_chain, Chain(WrappedFunction(_untuple), core_chain))
else
    Chain(embed_chain, core_chain)
end
```

`masked_input` decides whether `w` is forwarded into the assembled chain. Only
a padded group carries a mask, and grouped loaders shape `w` as `(1, 1, bs)`:

```
Chain:                               masked_input(model, x, w) = x
MaskedModel, w::Array{T,3} (grouped): masked_input(model, x, w) = (x, w)
MaskedModel, w::Vector (ungrouped):  masked_input(model, x, w) = x
```

Loss, eval, and infer all go through that hook. The grouped loader produces
`(x, y, w)`, or `(x, y, w, offset)` with `offset_name`.

Pad columns are still forwarded. The stream stays a rectangular
`(hidden, buffer)` matrix: BatchNorm and attention drop pads from reductions,
the head still emits a prediction per slot, and the loss zeros those slots.

## End-to-end flow

```
GroupedDataFrame
        │
        ▼
  padded buffers          x  (nfeats × buffer)
                        w/mask  (1 on real, 0/false on pad)
        │
        ▼
  masked_input(chain, x, w)
        │
        ├── Chain:          x                 → embed(x) → core(z)
        └── MaskedModel:    (x, w)            → embed(x, w) → core(z, w)
        │
        ▼
  y  (outsize × k × buffer) pad slots still have a number
        │
        ├── train/eval:  loss weighted by w  (pads drop out)
        └── infer:       pred[:, mask]       (pads dropped after forward)
```

## Inside MLPAttn / NeuroTreeAttn

`MLPAttn` converts `w` to a boolean `valid` once, then uses the same flags in
two shapes.

```
(x, w)
   │
   ├─ valid = vec(w) .> 0          length = buffer
   │
   ▼
encoder  ←  (x, valid)
   │
   │   residual MLP, same recipe as ResNet, but BN is masked:
   │
   │   CarryMask(Dense)     (h, valid) → (Dense(h), valid)
   │   MaskedBatchNorm      μ, σ from columns where valid is true
   │   CarryMask(Dropout)
   │   MaskSkip             (h, valid) → (h + f(h), valid)
   │
   ▼
z, drop the tuple
   │
   ▼
AttnResidual  ←  (z, mask)
   │
   │   mask = reshape(valid, buffer, 1, 1, 1)
   │   Q = K = W_qk z ,   V = z
   │   attention batch dim = 1  (the group is the sequence)
   │   rectangular key-padding mask (not causal):
   │   padded keys are zeroed for every query
   │
   ▼
head(z)        Dense on every column, including pads
```

`CarryMask` / `MaskSkip` exist so a `Chain` of ordinary layers can thread
`(features, mask)` without each `Dense` knowing about padding.
`MaskedBatchNorm` is the layer that actually reads the flag.

`NeuroTreeAttn` is the same split with a tree encoder: trees are
per-observation (`CarryMask` around `NeuroTree` + flatten), then
`MaskedBatchNorm` uses the valid-token flag like MLPAttn. Attention blocks
receive a rectangular key-padding mask. The hidden width is NeuroTree `k`
(one ensemble of `ntrees` per channel, scalar leaf preds), not the number of
leaves.

The mask is a key-padding mask over observations in the group, not a feature
mask and not a causal triangle.

## Ungrouped batches

With no `group_name`, there are no pad slots. Every column is a real
observation and every layer gets a plain `x`, even when the loader carries
sample weights: an unmasked `MaskedBatchNorm` is Lux's `BatchNorm`, so ungrouped
MLP and ResNet train exactly as with plain `BatchNorm`. `n_attn_layers=0` on
`MLPAttn` is that encoder plus a linear head: a ResNet ablation.

Grouped weights mark pads with `0`, so the grouped loader rejects non-positive
or non-finite weights.
