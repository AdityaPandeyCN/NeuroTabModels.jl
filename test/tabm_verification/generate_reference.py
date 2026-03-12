"""Generate reference outputs from official tabm + rtdl_num_embeddings.

Run:  python generate_reference.py
Produces: tabm_reference_data.npz
"""
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

B, K, D_IN, D_OUT, D_BLOCK = 8, 4, 10, 1, 32
N_BLOCKS = 2
D_EMB = 16
N_FREQ = 24
N_BINS = 8

x = torch.randn(B, D_IN)
x3d = x.unsqueeze(1).expand(-1, K, -1).clone()

data = {"x": x.numpy(), "x3d": x3d.numpy()}

# LinearBatchEnsemble 
# output: (B=8, K=4, D_BLOCK=32)

from tabm import LinearBatchEnsemble

lbe = LinearBatchEnsemble(D_IN, D_BLOCK, k=K, scaling_init="random-signs")
with torch.no_grad():
    y_lbe = lbe(x3d)
data["lbe_weight"] = lbe.weight.detach().numpy()
data["lbe_r"] = lbe.r.detach().numpy()
data["lbe_s"] = lbe.s.detach().numpy()
data["lbe_bias"] = lbe.bias.detach().numpy()
data["y_lbe"] = y_lbe.detach().numpy()

# LinearEnsemble
# output: (B=8, K=4, D_OUT=1)

from tabm import LinearEnsemble

le = LinearEnsemble(D_BLOCK, D_OUT, k=K)
le_input = torch.randn(B, K, D_BLOCK)
with torch.no_grad():
    y_le = le(le_input)
data["le_weight"] = le.weight.detach().numpy()
data["le_bias"] = le.bias.detach().numpy()
data["le_input"] = le_input.numpy()
data["y_le"] = y_le.detach().numpy()

# Full TabM variants
# output: (B=8, K=4, D_OUT=1)

from tabm import TabM

model = TabM.make(n_num_features=D_IN, d_out=D_OUT, k=K,
                  n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=0.0)
model.eval()
with torch.no_grad():
    y_tabm = model(x_num=x)
data["y_tabm"] = y_tabm.detach().numpy()
for name, param in model.named_parameters():
    data["tabm_" + name.replace(".", "_")] = param.detach().numpy()

model_mini = TabM.make(n_num_features=D_IN, d_out=D_OUT, k=K,
                       n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=0.0,
                       arch_type="tabm-mini")
model_mini.eval()
with torch.no_grad():
    y_mini = model_mini(x_num=x)
data["y_tabm_mini"] = y_mini.detach().numpy()
for name, param in model_mini.named_parameters():
    data["mini_" + name.replace(".", "_")] = param.detach().numpy()

model_packed = TabM.make(n_num_features=D_IN, d_out=D_OUT, k=K,
                         n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=0.0,
                         arch_type="tabm-packed")
model_packed.eval()
with torch.no_grad():
    y_packed = model_packed(x_num=x)
data["y_tabm_packed"] = y_packed.detach().numpy()
for name, param in model_packed.named_parameters():
    data["packed_" + name.replace(".", "_")] = param.detach().numpy()

# Numerical Embeddings

from rtdl_num_embeddings import (
    PiecewiseLinearEmbeddings,
    PeriodicEmbeddings,
    LinearEmbeddings,
    compute_bins,
)

x_emb_train = torch.randn(64, D_IN)  # need > N_BINS samples
x_emb = torch.randn(B, D_IN)
data["x_emb"] = x_emb.numpy()

# LinearEmbeddings
# params: weight (10, 16), bias (10, 16)
# output: (B=8, n_features=10, d_emb=16)

lin_emb = LinearEmbeddings(D_IN, D_EMB)
with torch.no_grad():
    y_lin_emb = lin_emb(x_emb)
data["lin_emb_weight"] = lin_emb.weight.detach().numpy()
data["lin_emb_bias"] = lin_emb.bias.detach().numpy()
data["y_lin_emb"] = y_lin_emb.detach().numpy()

# PeriodicEmbeddings
# params: periodic.weight (10, 24), linear.weight (10, 48, 16), linear.bias (10, 16)
# output: (B=8, n_features=10, d_emb=16)

per_emb = PeriodicEmbeddings(D_IN, D_EMB, n_frequencies=N_FREQ,
                              frequency_init_scale=0.01, lite=False)
with torch.no_grad():
    y_per_emb = per_emb(x_emb)
data["per_periodic_weight"] = per_emb.periodic.weight.detach().numpy()
data["per_linear_weight"] = per_emb.linear.weight.detach().numpy()
data["per_linear_bias"] = per_emb.linear.bias.detach().numpy()
data["y_per_emb"] = y_per_emb.detach().numpy()

# PiecewiseLinearEmbeddings (version B)
# params: linear0.weight (10, 16), linear0.bias (10, 16), linear.weight (10, 8, 16)
# output: (B=8, n_features=10, d_emb=16)

bins = compute_bins(x_emb_train, n_bins=N_BINS)
ple = PiecewiseLinearEmbeddings(bins, D_EMB, activation=False, version="B")
with torch.no_grad():
    y_ple = ple(x_emb)
data["ple_linear0_weight"] = ple.linear0.weight.detach().numpy()
data["ple_linear0_bias"] = ple.linear0.bias.detach().numpy()
data["ple_linear_weight"] = ple.linear.weight.detach().numpy()
data["y_ple"] = y_ple.detach().numpy()

# Save bins individually (list of 1D tensors, possibly ragged)
for i, b in enumerate(bins):
    data[f"ple_bins_{i}"] = b.numpy()
data["ple_n_features"] = np.array(len(bins))

# NOTE: Isolated PiecewiseLinearEncoding outputs flattened (B, total_bins)
# which differs from Julia's 3D (max_bins, n_features, batch).
# The full PLE end-to-end test covers encoding correctness implicitly.

# Full TabM† with PLE 
# TabM.make handles start_scaling_init automatically
# output: (B=8, K=4, D_OUT=1)

ple_for_tabm = PiecewiseLinearEmbeddings(bins, D_EMB, activation=False, version="B")
model_dagger = TabM.make(
    n_num_features=D_IN, d_out=D_OUT, k=K,
    n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=0.0,
    num_embeddings=ple_for_tabm,
)
model_dagger.eval()
with torch.no_grad():
    y_dagger = model_dagger(x_num=x)
data["y_tabm_dagger"] = y_dagger.detach().numpy()
for name, param in model_dagger.named_parameters():
    data["dagger_" + name.replace(".", "_")] = param.detach().numpy()

np.savez("tabm_reference_data.npz", **data)

print(f"Saved {len(data)} arrays to tabm_reference_data.npz")
print(f"  LBE:          {y_lbe.shape} mean={y_lbe.mean():.6f}")
print(f"  LE:           {y_le.shape} mean={y_le.mean():.6f}")
print(f"  TabM:         {y_tabm.shape} mean={y_tabm.mean():.6f}")
print(f"  TabM-mini:    {y_mini.shape} mean={y_mini.mean():.6f}")
print(f"  TabM-packed:  {y_packed.shape} mean={y_packed.mean():.6f}")
print(f"  LinearEmb:    {y_lin_emb.shape} mean={y_lin_emb.mean():.6f}")
print(f"  PeriodicEmb:  {y_per_emb.shape} mean={y_per_emb.mean():.6f}")
print(f"  PLE:          {y_ple.shape} mean={y_ple.mean():.6f}")
print(f"  TabM†:        {y_dagger.shape} mean={y_dagger.mean():.6f}")