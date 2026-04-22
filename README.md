# Fuse

A C++ ML compiler that takes a computation graph, fuses operators, and emits native x86/AArch64 machine code via LLVM IR and OrcJIT. The goal is to eliminate the two root causes of framework overhead on CPU inference — Python dispatch latency and intermediate tensor allocation.

---

## How it works

```
Python DSL → Graph IR → [ConstFold → MatmulEpilogue → VecLibSplit → Fusion → DCE → BufferReuse] → LLVMCodegen → OrcJIT → Runtime
```

**Key techniques:**

- **Operator fusion** — chains of element-wise ops (`add → relu → mul`) collapse into a single loop. LLVM O3 auto-vectorizes to AVX2 (8× float32/cycle) on x86 or NEON (4× float32/cycle) on AArch64.
- **Matmul epilogue fusion** — `Matmul → BiasAdd → activation` merges into a single `FusedMatmul` kernel: BLAS for the GEMM core, one fused loop for the trailing bias and activation.
- **Apple vecLib dispatch** *(macOS/ARM only)* — transcendental ops (`sigmoid`, `tanh`, `exp`, `log`) are routed through Apple's vForce library (`vvtanhf`, `vvexpf`, `vDSP_*`) instead of LLVM's generic scalar intrinsics. Hand-tuned ARM NEON implementations that process entire arrays at once.
- **Buffer reuse** — liveness analysis at compile time assigns buffer slots. A 5-op chain needs 2 slots, not 5. A 4-layer MLP needs 2 slots regardless of depth.
- **Zero heap on hot path** — `BufferPool` pre-allocates all intermediate buffers at compile time. Every inference call reuses the same pool; no `malloc` on the hot path.

---

## Benchmark results

> Full methodology and charts: [`results/benchmark-report.pdf`](results/benchmark-report.pdf)
>
> Run `python3 tests/test_correctness.py` before trusting any numbers. All 16 tests must pass.

**Platform:** Apple M-series · macOS · single-threaded CPU

### MLP inference — 4-layer, 256 hidden units, batch=1, ReLU

| Implementation | Latency (µs) | vs Fuse | Heap allocs | Peak memory |
|---|---|---|---|---|
| **Fuse** | **5.2** | — | **0** | **4 MB** |
| NumPy | 10.1 | 1.93× slower | ~3–5 per call | 12 MB |
| PyTorch eager | 10.9 | 2.09× slower | ~3–5 per call | 8 MB |
| torch.compile | 17.2 | 3.29× slower | ~3–5 per call | 8 MB |

Fuse wins on both axes: faster and less memory. One JIT'd function call per layer replaces 3 separate kernel dispatches through Python → C++ → BLAS → back. The `BufferPool` pre-allocates 2 slots and ping-pongs between them; NumPy allocates a fresh array on every operation.

`torch.compile` is the slowest: at batch=1 with 4 small layers, its dispatch overhead exceeds any fusion benefit it provides.

### Element-wise chain — add/relu/mul/sigmoid/tanh, N=1,000,000

| Implementation | Latency (ms) | vs Fuse |
|---|---|---|
| **Fuse** | **1.39** | — |
| PyTorch eager | 1.52 | 1.09× slower |
| NumPy | 2.69 | 1.94× slower |

On Apple Silicon, the `VecLibSplitPass` routes `sigmoid` and `tanh` through vForce (`vvtanhf`, `vvexpf` + `vDSP` arithmetic). The non-transcendental prefix (`add → relu → mul`) is fused into a single LLVM JIT kernel. Combined, this beats both NumPy and PyTorch.

### Memory — 5-op chain, N=1,000,000

| Implementation | Peak intermediate memory |
|---|---|
| NumPy | 12.0 MB |
| PyTorch | 8.0 MB |
| **Fuse** | **4.0 MB** |

NumPy: one allocation per op. PyTorch: caching allocator reuses blocks but old + new tensors coexist during each assignment — peak is 2 × 4 MB = 8 MB. Fuse: 2 pre-allocated slots, ping-ponged at compile time — 2× less than PyTorch, 3× less than NumPy.

### Fusion benefit vs chain length

The comparison vs PyTorch depends heavily on array size — two different regimes:

**N = 10,000 (cache-resident, dispatch-dominated)**

| Chain length | vs NumPy | vs PyTorch |
|---|---|---|
| 1 op | 0.73× — Fuse loses (JIT overhead, nothing to amortize) | 0.60× — Fuse loses |
| 2 ops | 1.75× | 1.22× |
| 3 ops | 2.12× | 1.64× |
| 5 ops | 3.12× | 2.75× |
| 8 ops | 4.95× | **7.13×** — PyTorch's per-op dispatch accumulates badly |

At N = 10K, PyTorch's per-op overhead compounds with chain length. At 8 ops, PyTorch (189 µs) is actually *slower* than NumPy (131 µs), while Fuse (27 µs) beats both by a wide margin.

**N = 5,000,000 (bandwidth-dominated)**

| Chain length | vs NumPy | vs PyTorch |
|---|---|---|
| 1 op | 0.65× — Fuse loses | 0.36× — Fuse loses |
| 3 ops | 1.78× | ~1.0× — tied |
| 5 ops | 1.78× | ~1.0× — tied |
| 8 ops | 1.84× | 0.87× — PyTorch wins |

At 5M elements (20 MB arrays), memory bandwidth is the ceiling. PyTorch's per-op Accelerate/MKL SIMD is very efficient at this scale and matches or beats Fuse. Fuse still consistently beats NumPy by 1.8× at chains ≥ 3 ops.

> Full data across all sizes: [`results/benchmark-report.pdf`](results/benchmark-report.pdf) §3

### Where Fuse wins and loses

| Workload | Winner | Note |
|---|---|---|
| MLP inference, batch=1 | **Fuse** | 1.93× vs NumPy, 2.09× vs PyTorch eager |
| Element-wise chains (add/relu/mul) | **Fuse** | 2.1–2.3× vs NumPy at N=10K |
| Chains with sigmoid/tanh (Apple Silicon) | **Fuse** | vForce beats PyTorch's MKL on ARM |
| Peak heap, any workload | **Fuse** | Pre-allocation always beats per-op malloc |
| Large-batch GEMM | PyTorch / NumPy | BLAS already optimal; epilogue fusion adds little |
| Arbitrary models (conv, attention, etc.) | PyTorch | Fuse supports only element-wise ops + matmul |

---

## Build

### macOS (Apple Silicon or Intel)

```bash
brew install llvm cmake python3
pip install pybind11 numpy matplotlib torch scikit-build-core

PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm \
      -DCMAKE_C_COMPILER=$(xcrun -f clang) \
      -DCMAKE_CXX_COMPILER=$(xcrun -f clang++) \
      -DCMAKE_CXX_FLAGS="-I$(brew --prefix llvm)/include" \
      -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

### Linux (Ubuntu 22.04)

```bash
# LLVM 19+ required for getOrInsertDeclaration
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main" | sudo tee /etc/apt/sources.list.d/llvm-19.list
sudo apt-get update -o Dir::Etc::sourcelist="sources.list.d/llvm-19.list" -o Dir::Etc::sourceparts="-"
sudo apt-get install -y llvm-19-dev libllvm19 clang-19 libopenblas-dev

pip install pybind11 numpy matplotlib torch scikit-build-core

PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=$(llvm-config-19 --cmakedir) \
      -DCMAKE_C_COMPILER=clang-19 \
      -DCMAKE_CXX_COMPILER=clang++-19 \
      -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build -j$(nproc)
```

---

## Reproducing benchmarks

Always run the correctness gate first.

```bash
python3 tests/test_correctness.py          # 16/16 must pass

python3 benchmarks/bench_elementwise.py    # element-wise vs NumPy + PyTorch
python3 benchmarks/bench_mlp.py            # MLP batch=1 vs NumPy + PyTorch + torch.compile
python3 benchmarks/bench_sweep.py          # sweep array sizes + chain lengths
python3 benchmarks/bench_memory.py         # peak memory comparison
python3 benchmarks/plot_results.py         # generate PNG charts from JSON
```

C++ unit tests:

```bash
cd build && ctest --output-on-failure
```

---

## Python API

```python
import fuse as ts
import numpy as np

# Element-wise fusion
g = ts.Graph()
a = g.input("a", shape=[1_000_000])
b = g.input("b", shape=[1_000_000])
x = g.add(a, b)
x = g.relu(x)
x = g.mul(x, b)
x = g.sigmoid(x)
x = g.tanh(x)
g.set_output(x)

fn = g.compile()   # runs all passes + codegen + JIT

a_np = np.random.randn(1_000_000).astype(np.float32)
b_np = np.random.randn(1_000_000).astype(np.float32)
out  = fn(a_np, b_np)   # executes the fused kernel

# MLP layer (matmul + bias + activation fused)
g = ts.Graph()
x = g.input("x", [1, 256])
W = g.input("W", [256, 256])
b = g.input("b", [256])
y = g.matmul(x, W)
y = g.bias_add(y, b)
y = g.relu(y)
g.set_output(y)
fn = g.compile()
```

---

## IR artifacts

After `g.compile(dump_ir=True)`, Fuse writes:

| File | Contents |
|---|---|
| `results/kernel_scalar.ll` | Unoptimized LLVM IR from codegen |
| `results/kernel_vectorized.ll` | Post-O3 IR — look for `<8 x float>` on x86 or `<4 x float>` on AArch64 |
| `results/ir_before_fusion.dot` | Graph before passes |
| `results/ir_after_fusion.dot` | Graph after passes |

---

## Architecture

| File | Role |
|---|---|
| `include/ts/ir.h` | `Node`, `Graph`, `TensorType`, `OpKind`, `FusedOp` |
| `include/ts/passes.h` | `PassBase` + all pass declarations |
| `src/passes/fusion.cpp` | Greedy element-wise chain fusion |
| `src/passes/matmul_epilogue.cpp` | `Matmul+BiasAdd+activation → FusedMatmul` |
| `src/passes/veclib_split.cpp` | Replace transcendentals with Apple vecLib dispatch (macOS only) |
| `src/passes/buffer_reuse.cpp` | Liveness-based buffer slot assignment |
| `src/codegen.cpp` | LLVM IR generation for `FusedKernel` and `FusedMatmul` |
| `src/jit.cpp` | OrcJIT module add + O3 optimization + symbol lookup |
| `src/runtime.cpp` | `BufferPool` + `CompiledFunction::execute` (BLAS dispatch + epilogue) |
| `bindings/python.cpp` | pybind11 module `fuse` |

---

## What this demonstrates

The techniques here — operator fusion, LLVM codegen, OrcJIT, buffer liveness reuse, matmul epilogue fusion — are the same techniques used in XLA, TVM, and Halide. This is a minimal but complete end-to-end implementation with benchmark evidence, demonstrating:

- C++ systems programming: IR ownership, raw pointer arithmetic, memory layout
- Compiler pipeline: IR design, pass infrastructure, codegen, JIT
- LLVM: OrcJIT, PassBuilder O3, loop vectorize metadata, intrinsics (`llvm.exp.f32`, `llvm.maxnum.f32`, etc.)
- Benchmark discipline: warmup, statistical reporting, correctness gating before timing
- Competitive honesty: knowing where BLAS wins (large GEMM) and where fusion wins (small-batch inference, element-wise chains)
