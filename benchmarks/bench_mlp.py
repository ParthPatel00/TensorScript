"""
Secondary benchmark: 4-layer 256-hidden MLP, batch=1.
Claims: 2-3x faster than PyTorch eager, competitive with torch.compile.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import fuse as ts

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HIDDEN  = 256
LAYERS  = 4
WARMUP  = 50
RUNS    = 500

rng = np.random.default_rng(42)

def randn(*shape):
    return rng.standard_normal(shape).astype(np.float32)

# ── Weights (fixed for the benchmark run) ─────────────────────────────────────
weights  = [randn(HIDDEN, HIDDEN) for _ in range(LAYERS)]
biases   = [randn(HIDDEN)         for _ in range(LAYERS)]
x_input  = randn(1, HIDDEN)

# ── Fuse graph ────────────────────────────────────────────────────────
# Register graph inputs in the same interleaved order as make_ts_inputs():
# x, W0, b0, W1, b1, ..., W3, b3
g   = ts.Graph()
x_h = g.input("x", [1, HIDDEN])

cur = x_h
for i in range(LAYERS):
    W_h = g.input(f"W{i}", [HIDDEN, HIDDEN])
    b_h = g.input(f"b{i}", [HIDDEN])
    cur = g.matmul(cur, W_h)
    cur = g.bias_add(cur, b_h)
    cur = g.relu(cur)
g.set_output(cur)

ts_fn = g.compile()

def bench(fn, *args):
    for _ in range(WARMUP): fn(*args)
    t0 = time.perf_counter()
    for _ in range(RUNS): fn(*args)
    return (time.perf_counter() - t0) / RUNS * 1_000  # ms

# ── Flatten inputs in the order the compiled function expects ─────────────────
# Graph inputs are: x, W0, b0, W1, b1, ..., W3, b3
def make_ts_inputs():
    args = [x_input]
    for i in range(LAYERS):
        args.append(weights[i])
        args.append(biases[i])
    return args

ts_inputs = make_ts_inputs()

# ── Correctness check against NumPy reference ────────────────────────────────
def numpy_mlp(x):
    for W, b in zip(weights, biases):
        x = np.maximum(0, x @ W + b)
    return x

ts_out  = ts_fn(*ts_inputs)
np_ref  = numpy_mlp(x_input).ravel()
assert np.allclose(ts_out, np_ref, atol=1e-4), \
    f"CORRECTNESS FAIL — max diff: {np.max(np.abs(ts_out - np_ref)):.6f}"
print("Correctness: PASS")

# ── Timing ────────────────────────────────────────────────────────────────────
results = {}

results["fuse"] = {"mean_ms": bench(ts_fn, *ts_inputs)}
results["numpy"]        = {"mean_ms": bench(numpy_mlp, x_input)}

if HAS_TORCH:
    w_t = [torch.from_numpy(W) for W in weights]
    b_t = [torch.from_numpy(b) for b in biases]
    x_t = torch.from_numpy(x_input)

    def torch_mlp_eager(x):
        for W, b in zip(w_t, b_t):
            x = torch.relu(x @ W + b)
        return x

    # Warmup torch
    torch_mlp_eager(x_t)
    results["pytorch_eager"] = {"mean_ms": bench(torch_mlp_eager, x_t)}

    try:
        torch_mlp_compiled = torch.compile(torch_mlp_eager, backend="inductor")
        torch_mlp_compiled(x_t)  # trigger compilation
        results["pytorch_compile"] = {"mean_ms": bench(torch_mlp_compiled, x_t)}
    except Exception as e:
        print(f"  torch.compile unavailable: {e}")

ts_ms = results["fuse"]["mean_ms"]
np_ms = results["numpy"]["mean_ms"]
results["speedup_vs_numpy"] = np_ms / ts_ms
if "pytorch_eager" in results:
    pt_ms = results["pytorch_eager"]["mean_ms"]
    results["speedup_vs_pytorch_eager"] = pt_ms / ts_ms
if "pytorch_compile" in results:
    ptc_ms = results["pytorch_compile"]["mean_ms"]
    results["speedup_vs_pytorch_compile"] = ptc_ms / ts_ms

results["config"] = {
    "hidden": HIDDEN, "layers": LAYERS,
    "batch": 1, "activation": "relu",
}

print(f"\n4-layer {HIDDEN}-hidden MLP (batch=1)")
print(f"  NumPy:           {np_ms:.4f} ms")
print(f"  Fuse:    {ts_ms:.4f} ms  ({np_ms/ts_ms:.2f}x vs NumPy)")
if "pytorch_eager" in results:
    print(f"  PyTorch eager:   {results['pytorch_eager']['mean_ms']:.4f} ms"
          f"  ({results['speedup_vs_pytorch_eager']:.2f}x vs PyTorch eager)")
if "pytorch_compile" in results:
    print(f"  torch.compile:   {results['pytorch_compile']['mean_ms']:.4f} ms"
          f"  ({results['speedup_vs_pytorch_compile']:.2f}x vs torch.compile)")

os.makedirs("results", exist_ok=True)
out_path = "results/bench_mlp.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults written to {out_path}")
