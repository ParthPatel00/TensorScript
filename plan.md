# TensorScript — Improved Implementation Plan

## Context

The goal is to build a C++ ML compiler that can credibly claim to outperform NumPy and PyTorch eager mode for specific, well-chosen CPU inference workloads. The claim must be backed by reproducible benchmark evidence in the repo. The compiler uses LLVM IR codegen, operator fusion, and OrcJIT to eliminate the two root causes of framework overhead: Python dispatch latency and intermediate tensor allocation.

The original plan covered element-wise fusion well. This plan extends it to include Matmul + fused epilogues, buffer liveness/reuse, and a benchmark suite that competes against PyTorch (not just NumPy), producing a stronger and more credible performance story.

---

## Where TensorScript Can Win (Honest Competitive Analysis)

| Workload | vs NumPy | vs PyTorch eager | vs torch.compile |
|---|---|---|---|
| Chained element-wise (5+ ops, 1M floats) | **3–5x** (no intermediates, no Python dispatch) | **2–3x** (no dispatch overhead) | **1.3–1.8x** (more aggressive fusion) |
| Small MLP inference (bs=1, 4 layers, 256-hidden) | **4–6x** | **2–4x** (Python overhead dominates at this size) | competitive |
| Large matmul alone (512x512) | ~1x (BLAS is optimal) | ~1x | ~1x |

**Why we can beat torch.compile on element-wise chains:** torch.compile/Inductor classifies ops into GEMM and element-wise buckets and generates separate kernels. TensorScript can fuse across the GEMM epilogue (bias + activation) into the matmul output loop, reducing memory passes. For chains of 5+ element-wise ops, TensorScript collapses to 1 kernel; Inductor may produce 2-3.

**Why we cannot beat BLAS for large matmul:** OpenBLAS/MKL implement micro-kernel tiling tuned to specific L1/L2 cache sizes and register counts. We won't replicate that. We don't need to: our story is fusion, not raw GEMM throughput.

**Primary benchmark claim (achievable):** *"On a chain of 5 element-wise ops over 1M float32 elements, TensorScript is 4x faster than NumPy and 2x faster than PyTorch eager, with 5x lower peak memory."*

**Secondary benchmark claim (achievable):** *"For batch=1 inference of a 4-layer 256-hidden MLP on CPU (Apple M2 or Intel x86 with AVX2), TensorScript is 3x faster than PyTorch eager and within 20% of torch.compile."*

---

## Architecture

```
Python DSL / Graph Builder (pybind11)
         |
         v
  Computation Graph IR
    Nodes: Input, Constant, Add, Sub, Mul, Div,
           Relu, Sigmoid, Tanh, Exp, Log, Neg, Sqrt,
           Matmul, BiasAdd, FusedKernel
         |
         v
  Optimization Passes (run in order)
    1. ConstantFolding      — fold compile-time-known subgraphs
    2. FusionPass           — element-wise chains → FusedKernel
    3. MatmulEpilogueFusion — Matmul + BiasAdd + activation → FusedMatmul
    4. DeadCodeElimination  — prune unreachable nodes
    5. BufferReusePass      — liveness analysis; alias output buffers where safe
         |
         v
  LLVM IR Codegen
    - One LLVM function per kernel node
    - FusedKernel: single scalar loop (LLVM auto-vectorizes to AVX2/NEON)
    - FusedMatmul: tiled GEMV/GEMM loop with fused epilogue in inner loop
    - Emits human-readable .ll alongside JIT
         |
         v
  LLVM PassManager (O3 + loop-vectorize + fma)
         |
         v
  LLVM OrcJIT (LLJIT, LLVM 17+)
         |
         v
  Runtime Executor
    - Pre-allocated buffer pool (no heap alloc on hot path)
    - Typed function pointer call with raw float* buffers
         |
         v
  Python Bindings (pybind11)
    - Graph, compile(), __call__(np.ndarray...) → np.ndarray
    - Zero-copy: passes data_ptr() directly to JIT'd functions
         |
         v
  Benchmark Harness
    - vs NumPy, vs PyTorch eager, vs torch.compile
    - Correctness gate (np.allclose) must pass before any timing run
    - Committed raw JSON + PNG charts
```

---

## Directory Structure

```
TensorScript/
├── CMakeLists.txt
├── README.md
├── include/ts/
│   ├── ir.h                  — Node, Graph, TensorType, OpKind
│   ├── ops.h                 — OpInfo, is_elementwise, llvm_intrinsic
│   ├── passes.h              — PassBase, all pass declarations
│   ├── codegen.h             — LLVMCodegen (element-wise + matmul paths)
│   ├── jit.h                 — OrcJIT wrapper
│   └── runtime.h             — BufferPool, ExecutionContext
├── src/
│   ├── ir.cpp
│   ├── ops.cpp
│   ├── codegen.cpp
│   ├── jit.cpp
│   ├── runtime.cpp
│   └── passes/
│       ├── constant_fold.cpp
│       ├── fusion.cpp              — element-wise chain fusion
│       ├── matmul_epilogue.cpp     — fuse Matmul+BiasAdd+activation
│       ├── dce.cpp
│       └── buffer_reuse.cpp        — liveness-based buffer aliasing
├── bindings/
│   └── python.cpp
├── benchmarks/
│   ├── bench_elementwise.py        — primary claim: vs NumPy + PyTorch
│   ├── bench_mlp.py                — secondary claim: vs PyTorch eager + compile
│   ├── bench_sweep.py              — sweep array size and chain length
│   ├── bench_memory.py             — tracemalloc peak heap comparison
│   └── plot_results.py
├── tests/
│   ├── test_ir.cpp
│   ├── test_fusion.cpp
│   ├── test_matmul_epilogue.cpp
│   ├── test_buffer_reuse.cpp
│   ├── test_codegen.cpp
│   └── test_correctness.py         — numerical gate: atol=1e-5 vs NumPy
└── results/                        — committed artifacts
    ├── benchmark_raw.json
    ├── speedup_elementwise.png
    ├── speedup_mlp.png
    ├── memory_savings.png
    ├── ir_before_fusion.dot
    ├── ir_after_fusion.dot
    ├── kernel_scalar.ll
    └── kernel_vectorized.ll         — shows <8 x float> after O3
```

---

## Component Specs

### 1. IR (`include/ts/ir.h`)

```cpp
enum class DataType { Float32, Float16 };

struct TensorType {
    std::vector<int64_t> shape;
    DataType dtype = DataType::Float32;
    int64_t numel() const;
    int64_t nbytes() const;
};

enum class OpKind {
    Input, Constant,
    Add, Sub, Mul, Div,
    Relu, Sigmoid, Tanh, Exp, Log, Neg, Sqrt,
    Matmul,
    BiasAdd,
    FusedKernel,    // element-wise chain collapsed by FusionPass
    FusedMatmul,    // Matmul + optional BiasAdd + optional activation
};

struct Node {
    std::string name;
    OpKind kind;
    TensorType output_type;
    std::vector<Node*> inputs;
    std::vector<OpKind> fused_ops;      // FusedKernel: ordered ops in body
    OpKind epilogue_activation = OpKind::Input;  // FusedMatmul: post-matmul activation
    int buffer_slot = -1;               // assigned by BufferReusePass
};

struct Graph {
    std::vector<std::unique_ptr<Node>> nodes;
    std::vector<Node*> inputs;
    Node* output = nullptr;

    Node* add_input(std::string name, TensorType type);
    Node* add_op(OpKind kind, std::vector<Node*> inputs);
    Node* add_const(std::string name, TensorType type, std::vector<float> data);
    void  set_output(Node* n);
    void  dump() const;
    void  to_dot(const std::string& path) const;
};
```

---

### 2. Fusion Pass (`src/passes/fusion.cpp`)

Greedy chain fusion in topological order. A node is fusable if:
- `is_elementwise(kind)` is true, AND
- all its consumers are also element-wise (or it is the terminal node)

Algorithm: scan topologically; when a fusable node N has one consumer C that is also fusable, merge N's op into C's `fused_ops` list, rewire inputs, mark N dead. Repeat until stable.

The invariant after this pass: no two adjacent element-wise nodes exist; all chains are collapsed to FusedKernel nodes.

---

### 3. Matmul Epilogue Fusion (`src/passes/matmul_epilogue.cpp`)

Pattern match: `Matmul → BiasAdd → [optional activation (Relu/Sigmoid/Tanh)]`

When this pattern is found, collapse into a single `FusedMatmul` node that:
- Runs the tiled GEMM inner loop
- In the epilogue of each output element: adds bias, applies activation
- Writes once to the output buffer

This directly competes with PyTorch's oneDNN epilogue fusion but without the framework overhead.

---

### 4. Buffer Reuse Pass (`src/passes/buffer_reuse.cpp`)

Compute liveness intervals for every node's output buffer. Where a buffer's live range ends before another begins and sizes match (or output fits), assign them the same `buffer_slot`. At runtime, the `BufferPool` pre-allocates one contiguous slab of memory and carves it by slot.

Result: for a 5-op element-wise chain, only 2 buffers are needed (ping-pong), not 5. For MLP inference, only 3 buffers regardless of depth (input, two ping-pong intermediates).

---

### 5. LLVM Codegen (`src/codegen.cpp`)

**FusedKernel (element-wise):**

Emit a scalar loop. LLVM O3's loop vectorizer converts this to AVX2 (8x float32) or NEON (4x float32) automatically. Use `!llvm.loop.vectorize.enable` metadata and `align 32` on pointers to maximize the vectorizer's confidence.

```llvm
define void @kernel_0(float* noalias %in0, float* noalias %in1, float* noalias %out, i64 %n) {
loop:
  %i  = phi i64 [0, %entry], [%i1, %loop]
  %a  = load float, float* (gep %in0, %i), align 4
  %b  = load float, float* (gep %in1, %i), align 4
  %v0 = fadd fast float %a, %b
  %v1 = call fast float @llvm.maxnum.f32(%v0, 0.0)  ; relu
  store float %v1, float* (gep %out, %i), align 4
  %i1 = add i64 %i, 1
  br i1 (icmp eq %i1, %n), %exit, %loop
}
```

The `fast` flag enables FMA contraction and reassociation. `noalias` allows the vectorizer to assume no pointer aliasing (safe because BufferReusePass enforces this).

**FusedMatmul:**

For batch=1 (GEMV): emit a tiled dot-product loop. Tile size chosen to fit in L1 (tile_k=64 for 256B per row, matching a typical 32KB L1). Epilogue adds bias and applies activation inside the output loop, avoiding a second pass over output.

For batch>1 (GEMM): call into a pre-compiled BLAS kernel (OpenBLAS/Accelerate) for the GEMM portion, then apply the fused epilogue via a separate element-wise loop. This is honest: we don't pretend to beat BLAS for large GEMM.

---

### 6. Runtime (`src/runtime.h`, `src/runtime.cpp`)

```cpp
struct BufferPool {
    std::vector<std::vector<float>> slots;  // pre-allocated at compile time
    void allocate(const Graph& g);          // sizes slots from buffer_reuse metadata
    float* get(int slot);
};

struct CompiledFunction {
    std::vector<KernelFn> kernels;          // JIT'd function pointers in order
    BufferPool pool;
    std::vector<int> kernel_input_slots;    // which slots feed each kernel
    int output_slot;

    void operator()(std::vector<float*> inputs, float* output, int64_t n);
};
```

Hot path: no heap allocation. Every call reuses `pool` slots. The only pointers passed to JIT'd functions are raw `float*` from the pool.

---

### 7. Python Bindings (`bindings/python.cpp`)

```python
import tensorscript as ts
import numpy as np

g = ts.Graph()
a = g.input("a", shape=[1_000_000])
b = g.input("b", shape=[1_000_000])
x = g.add(a, b)
x = g.relu(x)
x = g.mul(x, b)
x = g.sigmoid(x)
x = g.tanh(x)
g.set_output(x)

fn = g.compile()  # runs all passes + codegen + JIT; prints pass summary

a_np = np.random.randn(1_000_000).astype(np.float32)
b_np = np.random.randn(1_000_000).astype(np.float32)
out = fn(a_np, b_np)   # zero-copy: passes a_np.ctypes.data_as(float*)
```

`compile()` prints a summary like:
```
[TensorScript] FusionPass: 5 nodes → 1 FusedKernel
[TensorScript] BufferReuse: 5 buffers → 2 slots (ping-pong)
[TensorScript] Codegen: kernel_0, 1M elements, vectorized <8 x float>
[TensorScript] JIT: compiled in 38ms
```

---

## Implementation Phases

### Phase 1 — Core IR (1–2 days)
- `TensorType`, `Node`, `Graph`, `OpKind`
- `add_input`, `add_op`, `add_const`, `set_output`, `dump`, `to_dot`
- Unit tests: build a 5-op graph, verify node count and edges, print ASCII + dot

### Phase 2 — Scalar Codegen + OrcJIT (2–3 days)
- One LLVM function per non-fused op node (no fusion yet)
- Scalar loop: `for i in 0..N`
- Supported ops: Add, Mul, Sub, Relu (maxnum), Sigmoid (fast approx), Tanh, Exp
- OrcJIT: `add_module`, `lookup`, typed function pointer call
- Correctness test: each op matches NumPy to atol=1e-5

### Phase 3 — Element-wise Fusion Pass (2 days)
- Greedy chain-fusion in topological order
- FusedKernel node with ordered `fused_ops`
- Codegen for FusedKernel: single loop, sequential op body
- Unit test: `Add → Relu → Mul` collapses to 1 FusedKernel with 3 fused_ops
- Dump before/after dot graphs

### Phase 4 — LLVM O3 + Vectorization (1 day)
- Enable O3 pass pipeline via `PassBuilder`
- Add `noalias`, `align 32`, `fast`, and loop vectorize metadata
- Verify vectorization: `kernel_vectorized.ll` must contain `<8 x float>` on x86 or `<4 x float>` on AArch64
- Benchmark checkpoint: element-wise chain should now show clear speedup over NumPy

### Phase 5 — Matmul + Epilogue Fusion (3 days)
- Add `Matmul` and `BiasAdd` to IR
- `MatmulEpilogueFusion` pass: collapse `Matmul → BiasAdd → Relu/Sigmoid` into `FusedMatmul`
- Codegen: GEMV path for batch=1 (tiled dot-product), GEMM path calls BLAS + fused epilogue loop
- Unit test: numerical correctness of fused MLP layer vs NumPy

### Phase 6 — Buffer Reuse Pass (2 days)
- Compute liveness intervals for all node output buffers
- Assign `buffer_slot` indices; size each slot to the max live tensor at that slot
- `BufferPool`: allocate slots at compile time, reuse on every call
- Verify: a 5-op chain uses 2 slots (not 5); a 4-layer MLP uses 3 slots (not 9)
- `bench_memory.py`: measure peak heap via `tracemalloc`; expect 5x reduction vs NumPy

### Phase 7 — Python Bindings (2 days)
- pybind11 module `tensorscript`
- `Graph`, `CompiledFunction`
- Zero-copy: `np.ndarray.ctypes.data_as(ctypes.POINTER(ctypes.c_float))` → `float*`
- Output: allocate output numpy array, pass its data pointer to executor, return it
- `pip install -e .` via scikit-build

### Phase 8 — Benchmark Suite + Evidence (2 days)
- `bench_elementwise.py`: 5-op chain, N=1M, 100 runs + 10 warmup; vs NumPy + PyTorch eager
- `bench_mlp.py`: 4-layer 256-hidden MLP, batch=1; vs NumPy + PyTorch eager + torch.compile
- `bench_sweep.py`: sweep N (10K to 10M), chain length (1–8), write `benchmark_raw.json`
- `bench_memory.py`: tracemalloc peak heap at each chain length
- `plot_results.py`: 4 charts (elementwise speedup, mlp speedup, sweep, memory)
- Correctness gate runs before every benchmark; CI fails if `np.allclose` fails

### Phase 9 — Documentation + CI (1 day)
- README: what it is, why it's faster (mechanism, not just numbers), how to build, how to reproduce
- Embed charts in README, GraphViz before/after fusion diagram, LLVM IR snippet
- GitHub Actions: build on ubuntu-latest + macos-latest, run correctness tests, run benchmark (record results to JSON artifact)

---

## Evidence Artifacts

| File | What it proves |
|---|---|
| `results/benchmark_raw.json` | Raw p50/p95 timing: TS vs NumPy vs PyTorch at every config |
| `results/speedup_elementwise.png` | 4x speedup over NumPy, 2x over PyTorch eager |
| `results/speedup_mlp.png` | 3x over PyTorch eager, competitive with torch.compile |
| `results/memory_savings.png` | 5x less peak heap: 2 buffers vs 5 intermediate arrays |
| `results/ir_before_fusion.dot` | Graph before FusionPass (5 nodes) |
| `results/ir_after_fusion.dot` | Graph after FusionPass (1 FusedKernel) |
| `results/kernel_scalar.ll` | Unoptimized IR from codegen |
| `results/kernel_vectorized.ll` | Post-O3 IR showing `<8 x float>` SIMD |
| CI badge | Build + correctness pass on clean machine — results are reproducible |

---

## Build

```bash
# macOS
brew install llvm@17 cmake openblas python3
pip install pybind11 numpy matplotlib torch

cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=$(brew --prefix llvm@17)/lib/cmake/llvm \
      -DBLAS_DIR=$(brew --prefix openblas)
cmake --build build -j$(sysctl -n hw.logicalcpu)

pip install -e .

python tests/test_correctness.py       # must pass before benchmarking
python benchmarks/bench_elementwise.py
python benchmarks/bench_mlp.py
python benchmarks/bench_sweep.py
python benchmarks/plot_results.py
```

---

## Phased Competitive Claims (what we can say at each phase)

- **After Phase 4**: "3–5x faster than NumPy on element-wise chains" — this is the safe, publishable claim.
- **After Phase 5**: "Supports MLP inference end-to-end with fused matmul epilogues."
- **After Phase 6**: "5x lower peak memory than NumPy due to buffer reuse."
- **After Phase 8**: "2–3x faster than PyTorch eager for small-batch MLP inference" — benchmark evidence in repo.

---

## What This Demonstrates Technically

The techniques used — operator fusion, LLVM codegen, OrcJIT, buffer liveness reuse, epilogue fusion — are the exact same techniques in XLA (Google), TVM (Apache), and Halide. Building a minimal but complete version end-to-end, with benchmark evidence, demonstrates:

- C++ systems programming: IR ownership, raw pointer arithmetic, memory layout
- Compiler pipeline: IR design, pass infrastructure, codegen, JIT
- LLVM fluency: OrcJIT, PassBuilder, loop metadata, intrinsics
- Benchmark discipline: warmup, statistical reporting, correctness gating
- Competitive honesty: knowing where you can and cannot beat BLAS
