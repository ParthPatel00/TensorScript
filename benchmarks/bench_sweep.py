"""
Sweep array size and chain length to produce the full speedup dataset.
Compares Fuse, NumPy, and PyTorch eager at every (N, chain_length) combination.
Writes results/benchmark_raw.json.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import torch
import fuse as ts

WARMUP = 10
RUNS = 100

rng = np.random.default_rng(1)

OP_SEQUENCE = ["add", "relu", "mul", "sigmoid", "tanh", "neg", "exp", "sqrt"]

NP_OPS = {
    "add":     lambda acc, b: acc + b,
    "sub":     lambda acc, b: acc - b,
    "mul":     lambda acc, b: acc * b,
    "relu":    lambda acc, b: np.maximum(acc, 0),
    "sigmoid": lambda acc, b: 1 / (1 + np.exp(-acc)),
    "tanh":    lambda acc, b: np.tanh(acc),
    "neg":     lambda acc, b: -acc,
    "exp":     lambda acc, b: np.exp(np.clip(acc, -10, 10)),
    "sqrt":    lambda acc, b: np.sqrt(np.abs(acc) + 0.001),
}

PT_OPS = {
    "add":     lambda acc, b: acc + b,
    "sub":     lambda acc, b: acc - b,
    "mul":     lambda acc, b: acc * b,
    "relu":    lambda acc, b: torch.relu(acc),
    "sigmoid": lambda acc, b: torch.sigmoid(acc),
    "tanh":    lambda acc, b: torch.tanh(acc),
    "neg":     lambda acc, b: -acc,
    "exp":     lambda acc, b: torch.exp(acc.clamp(-10, 10)),
    "sqrt":    lambda acc, b: torch.sqrt(acc.abs() + 0.001),
}

def bench(fn, *args, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup): fn(*args)
    t0 = time.perf_counter()
    for _ in range(runs): fn(*args)
    return (time.perf_counter() - t0) / runs * 1000

SIZES   = [10_000, 100_000, 500_000, 1_000_000, 5_000_000]
LENGTHS = [1, 2, 3, 5, 8]

all_results = []
total = len(SIZES) * len(LENGTHS)
done = 0

for N in SIZES:
    a_np = rng.standard_normal(N).astype(np.float32)
    b_np = rng.standard_normal(N).astype(np.float32)
    a_np = np.clip(a_np, -5, 5); b_np = np.clip(b_np, -5, 5)
    a_t  = torch.from_numpy(a_np)
    b_t  = torch.from_numpy(b_np)

    for L in LENGTHS:
        ops = OP_SEQUENCE[:L]
        done += 1
        print(f"[{done}/{total}] N={N:>9,} L={L} ops={ops}", flush=True)

        # NumPy
        def numpy_fn(a, b, _ops=ops):
            acc = a.copy()
            for op in _ops:
                acc = NP_OPS[op](acc, b)
            return acc

        # PyTorch eager
        def pytorch_fn(a_t, b_t, _ops=ops):
            acc = a_t
            with torch.no_grad():
                for op in _ops:
                    acc = PT_OPS[op](acc, b_t)
            return acc

        # Fuse
        g = ts.Graph()
        ah = g.input("a", [N]); bh = g.input("b", [N])
        x = ah
        for op in ops:
            if op in ("add", "sub", "mul"):
                x = getattr(g, op)(x, bh)
            else:
                x = getattr(g, op)(x)
        g.set_output(x)
        ts_fn = g.compile()

        np_ms = bench(numpy_fn, a_np, b_np)
        pt_ms = bench(pytorch_fn, a_t, b_t)
        fs_ms = bench(ts_fn, a_np, b_np)

        all_results.append({
            "N": N, "chain_length": L, "ops": ops,
            "numpy_ms":   np_ms,
            "pytorch_ms": pt_ms,
            "fuse_ms":    fs_ms,
            "speedup_vs_numpy":   np_ms / fs_ms,
            "speedup_vs_pytorch": pt_ms / fs_ms,
        })

os.makedirs("results", exist_ok=True)
with open("results/benchmark_raw.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nWrote results/benchmark_raw.json")
