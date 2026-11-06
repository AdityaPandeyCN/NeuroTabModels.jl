"""
    _BatchNormEmbeddings(; nfeats)

Feature-wise `BatchNorm` on `(nfeats, batch)` input, or on `(x, w)` from a padded group so
pads stay out of the statistics. Output shape matches input.

# Arguments
- `nfeats::Int`: Number of input features.
"""
struct _BatchNormEmbeddings{L} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer::L
end

_BatchNormEmbeddings(; nfeats::Int) = _BatchNormEmbeddings(MaskedBatchNorm(nfeats))

(l::_BatchNormEmbeddings)(x, ps, st) = l.layer(x, ps, st)

LuxCore.outputsize(l::_BatchNormEmbeddings, x, ::AbstractRNG) = (size(x, 1),)