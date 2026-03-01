"""python generate_reference.py → tabm_reference_data.npz"""
import numpy as np, torch
from tabm import LinearBatchEnsemble, LinearEnsemble, TabM

torch.manual_seed(42)
B, K, D_IN, D_OUT, D_BLOCK, N_BLOCKS = 8, 4, 10, 1, 32, 2

x = torch.randn(B, D_IN)
x3d = x.unsqueeze(1).expand(-1, K, -1).clone()
data = {"x": x.numpy(), "x3d": x3d.numpy()}

# Layer: LinearBatchEnsemble
lbe = LinearBatchEnsemble(D_IN, D_BLOCK, k=K, scaling_init="random-signs")
with torch.no_grad():
    data["y_lbe"] = lbe(x3d).numpy()
for n in ("weight", "r", "s", "bias"):
    data[f"lbe_{n}"] = getattr(lbe, n).detach().numpy()

# Layer: LinearEnsemble
le = LinearEnsemble(D_BLOCK, D_OUT, k=K)
le_in = torch.randn(B, K, D_BLOCK)
with torch.no_grad():
    data["y_le"] = le(le_in).numpy()
data["le_input"] = le_in.numpy()
for n in ("weight", "bias"):
    data[f"le_{n}"] = getattr(le, n).detach().numpy()

# Full models
for arch, prefix in [("tabm", "tabm_"), ("tabm-mini", "mini_"), ("tabm-packed", "packed_")]:
    m = TabM.make(n_num_features=D_IN, d_out=D_OUT, k=K,
                  n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=0.0, arch_type=arch)
    m.eval()
    with torch.no_grad():
        data[f"y_{prefix.rstrip('_')}"] = m(x_num=x).numpy()
    for name, param in m.named_parameters():
        data[prefix + name.replace(".", "_")] = param.detach().numpy()

np.savez("tabm_reference_data.npz", **data)
print(f"Saved {len(data)} arrays")