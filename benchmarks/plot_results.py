"""
Generate benchmark charts from results/benchmark_raw.json and bench_elementwise.json.
Writes PNG files to results/.
"""
import json, os
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("matplotlib not installed — skipping plots")
    raise SystemExit(0)

os.makedirs("results", exist_ok=True)

# ── 1. Speedup vs chain length (at N=1M) ─────────────────────────────────────
if os.path.exists("results/benchmark_raw.json"):
    with open("results/benchmark_raw.json") as f:
        raw = json.load(f)

    N_TARGET = 1_000_000
    rows = [r for r in raw if r["N"] == N_TARGET]
    rows.sort(key=lambda r: r["chain_length"])

    lengths  = [r["chain_length"] for r in rows]
    speedups = [r["speedup"]      for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(lengths, speedups, color="#4C72B0", width=0.6)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Chain length (number of fused ops)")
    ax.set_ylabel("Speedup over NumPy")
    ax.set_title(f"Fuse vs NumPy — Element-wise chain (N={N_TARGET:,})")
    ax.set_xticks(lengths)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    for i, (l, s) in enumerate(zip(lengths, speedups)):
        ax.text(l, s + 0.05, f"{s:.1f}x", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/speedup_vs_chainlen.png", dpi=150)
    plt.close()
    print("Wrote results/speedup_vs_chainlen.png")

    # ── 2. Speedup vs array size (at chain length 5) ──────────────────────────
    L_TARGET = 5
    rows2 = [r for r in raw if r["chain_length"] == L_TARGET]
    rows2.sort(key=lambda r: r["N"])
    sizes    = [r["N"]      for r in rows2]
    speedups2 = [r["speedup"] for r in rows2]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(sizes, speedups2, marker="o", color="#55A868", linewidth=2)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Array size (N)")
    ax.set_ylabel("Speedup over NumPy")
    ax.set_title(f"Fuse vs NumPy — Array size sweep (chain={L_TARGET})")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    plt.savefig("results/speedup_vs_arraysize.png", dpi=150)
    plt.close()
    print("Wrote results/speedup_vs_arraysize.png")

# ── 3. Memory comparison ──────────────────────────────────────────────────────
if os.path.exists("results/bench_memory.json"):
    with open("results/bench_memory.json") as f:
        mem = json.load(f)

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["NumPy", "Fuse"]
    values = [mem["numpy_peak_mb"], mem["fuse_peak_mb"]]
    colors = ["#C44E52", "#4C72B0"]
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.1f} MB",
                ha="center", fontsize=10)
    ax.set_ylabel("Peak heap usage (MB)")
    ax.set_title(f"Peak memory — 5-op chain, N={mem['N']:,}")
    plt.tight_layout()
    plt.savefig("results/memory_savings.png", dpi=150)
    plt.close()
    print("Wrote results/memory_savings.png")

print("Plot generation complete.")
