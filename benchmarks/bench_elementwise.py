"""
Primary benchmark: chained element-wise ops.
Claims: 3-5x faster than NumPy, 2-3x faster than PyTorch eager.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import tensorscript as ts

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

N = 1_000_000
WARMUP = 20
RUNS = 200

def bench(fn, *args):
    for _ in range(WARMUP): fn(*args)
    t0 = time.perf_counter()
    for _ in range(RUNS): fn(*args)
    return (time.perf_counter() - t0) / RUNS * 1000  # ms

rng = np.random.default_rng(0)
a_np = np.clip(rng.standard_normal(N).astype(np.float32), -5, 5)
b_np = np.clip(rng.standard_normal(N).astype(np.float32), -5, 5)

# ── NumPy baseline ────────────────────────────────────────────────────────────
def numpy_chain(a, b):
    x = a + b
    x = np.maximum(x, 0)
    x = x * b
    x = 1 / (1 + np.exp(-x))
    return np.tanh(x)

# ── TensorScript ──────────────────────────────────────────────────────────────
g = ts.Graph()
a_h = g.input("a", [N])
b_h = g.input("b", [N])
x = g.add(a_h, b_h)
x = g.relu(x)
x = g.mul(x, b_h)
x = g.sigmoid(x)
x = g.tanh(x)
g.set_output(x)
ts_fn = g.compile()

# ── PyTorch eager ─────────────────────────────────────────────────────────────
if HAS_TORCH:
    a_t = torch.from_numpy(a_np)
    b_t = torch.from_numpy(b_np)
    def torch_chain(a, b):
        x = a + b
        x = torch.relu(x)
        x = x * b
        x = torch.sigmoid(x)
        return torch.tanh(x)

# ── Correctness check ─────────────────────────────────────────────────────────
ts_out  = ts_fn(a_np, b_np)
np_ref  = numpy_chain(a_np, b_np)
assert np.allclose(ts_out, np_ref, atol=1e-5), "CORRECTNESS FAIL — aborting benchmark"
print("Correctness: PASS")

# ── Timing ────────────────────────────────────────────────────────────────────
results = {}
results["numpy"]         = {"mean_ms": bench(numpy_chain, a_np, b_np)}
results["tensorscript"]  = {"mean_ms": bench(ts_fn, a_np, b_np)}
if HAS_TORCH:
    results["pytorch_eager"] = {"mean_ms": bench(torch_chain, a_t, b_t)}

results["config"] = {"N": N, "chain_length": 5, "ops": ["add","relu","mul","sigmoid","tanh"]}

ts_ms  = results["tensorscript"]["mean_ms"]
np_ms  = results["numpy"]["mean_ms"]
results["speedup_vs_numpy"] = np_ms / ts_ms
if HAS_TORCH:
    pt_ms = results["pytorch_eager"]["mean_ms"]
    results["speedup_vs_pytorch"] = pt_ms / ts_ms

print(f"\nElement-wise chain (N={N:,}, 5 ops)")
print(f"  NumPy:        {np_ms:.3f} ms")
print(f"  TensorScript: {ts_ms:.3f} ms  ({np_ms/ts_ms:.2f}x vs NumPy)")
if HAS_TORCH:
    print(f"  PyTorch:      {pt_ms:.3f} ms  ({pt_ms/ts_ms:.2f}x vs PyTorch)")

os.makedirs("results", exist_ok=True)
out_path = "results/bench_elementwise.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults written to {out_path}")
