"""python generate_embedding_reference.py → embedding_reference_data.npz"""
import numpy as np, torch
from rtdl_num_embeddings import (
    LinearEmbeddings, LinearReLUEmbeddings, PeriodicEmbeddings,
    PiecewiseLinearEmbeddings, compute_bins,
)

torch.manual_seed(42)
B, N_FEAT, D_EMB = 8, 5, 12
x = torch.randn(B, N_FEAT)
data = {"x": x.numpy()}

# 1. LinearEmbeddings
le = LinearEmbeddings(N_FEAT, D_EMB)
with torch.no_grad():
    data["y_linear"] = le(x).numpy()
data["linear_weight"] = le.weight.detach().numpy()
data["linear_bias"] = le.bias.detach().numpy()

# 2. LinearReLUEmbeddings
lre = LinearReLUEmbeddings(N_FEAT, D_EMB)
with torch.no_grad():
    data["y_linear_relu"] = lre(x).numpy()
data["linear_relu_weight"] = lre.linear.weight.detach().numpy()
data["linear_relu_bias"] = lre.linear.bias.detach().numpy()

# 3. PeriodicEmbeddings (lite=False)
pe = PeriodicEmbeddings(N_FEAT, D_EMB, n_frequencies=16, frequency_init_scale=0.01, lite=False)
with torch.no_grad():
    data["y_periodic"] = pe(x).numpy()
data["periodic_freq"] = pe.periodic.weight.detach().numpy()
data["periodic_linear_weight"] = pe.linear.weight.detach().numpy()
data["periodic_linear_bias"] = pe.linear.bias.detach().numpy()

# 4. PeriodicEmbeddings (lite=True)
pe_lite = PeriodicEmbeddings(N_FEAT, D_EMB, n_frequencies=16, frequency_init_scale=0.01, lite=True)
with torch.no_grad():
    data["y_periodic_lite"] = pe_lite(x).numpy()
data["periodic_lite_freq"] = pe_lite.periodic.weight.detach().numpy()
data["periodic_lite_linear_weight"] = pe_lite.linear.weight.detach().numpy()
data["periodic_lite_linear_bias"] = pe_lite.linear.bias.detach().numpy()

# 5. PiecewiseLinearEmbeddings (version B)
bins = compute_bins(torch.randn(500, N_FEAT), n_bins=16)
ple_b = PiecewiseLinearEmbeddings(bins, D_EMB, activation=False, version='B')
with torch.no_grad():
    data["y_ple_b"] = ple_b(x).numpy()
data["ple_b_linear0_weight"] = ple_b.linear0.weight.detach().numpy()
data["ple_b_linear0_bias"] = ple_b.linear0.bias.detach().numpy()
data["ple_b_linear_weight"] = ple_b.linear.weight.detach().numpy()
# Store bins for Julia
for i, b in enumerate(bins):
    data[f"bin_{i}"] = b.numpy()
data["n_bin_features"] = np.array(N_FEAT)

# 6. PiecewiseLinearEmbeddings (version A)
ple_a = PiecewiseLinearEmbeddings(bins, D_EMB, activation=False, version='A')
with torch.no_grad():
    data["y_ple_a"] = ple_a(x).numpy()
data["ple_a_linear_weight"] = ple_a.linear.weight.detach().numpy()
data["ple_a_linear_bias"] = ple_a.linear.bias.detach().numpy()

np.savez("embedding_reference_data.npz", **data)
print(f"Saved {len(data)} arrays")