# TensorScript — Project Guide for Claude

## What This Is

A C++ ML compiler that takes a computation graph, fuses operators, and emits native x86/AArch64 machine code via LLVM IR + OrcJIT. The goal is to provably outperform NumPy and PyTorch eager mode on specific CPU inference workloads.

## Credible Performance Claims

- **Primary**: 3–5x faster than NumPy on chained element-wise ops (5+ ops, 1M floats)
- **Secondary**: 2–3x faster than PyTorch eager for small-batch MLP inference (bs=1)
- **Memory**: 5x lower peak heap via buffer reuse (ping-pong allocation)
- **We do NOT claim to beat BLAS for large GEMM** — we use cblas for the GEMM core and fuse only the epilogue

## Architecture Summary

```
Python DSL → Graph IR → [ConstFold → FusionPass → MatmulEpilogueFusion → DCE → BufferReuse] → LLVMCodegen → OrcJIT → Runtime
```

## Key Design Decisions

- **FusedKernel**: scalar loop emitted as LLVM IR; LLVM O3 auto-vectorizes to AVX2/NEON
- **FusedMatmul**: cblas_sgemv/sgemm for GEMM core + fused epilogue (bias + activation) in element-wise loop
- **BufferPool**: all intermediate buffers pre-allocated at compile time; zero heap alloc on hot path
- **Kernel signature**: `void kernel_N(float** inputs, int num_inputs, float* output, int64_t n)`
- **noalias**: all pointer args declared noalias (BufferReusePass enforces non-aliasing)

## FusedOp Data Model

For a chain `a + b → relu → * b → sigmoid`, the FusedKernel has:
- inputs: [a, b]
- acc = inputs[0][i]  (init)
- FusedOp{Add, 1}: acc = acc + inputs[1][i]
- FusedOp{Relu, -1}: acc = relu(acc)  (input_idx=-1 means unary)
- FusedOp{Mul, 1}: acc = acc * inputs[1][i]
- FusedOp{Sigmoid, -1}: acc = sigmoid(acc)
- out[i] = acc

## Build

```bash
# macOS
brew install llvm@17 cmake openblas python3
pip install pybind11 numpy matplotlib torch scikit-build-core

cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=$(brew --prefix llvm@17)/lib/cmake/llvm
cmake --build build -j$(sysctl -n hw.logicalcpu)
pip install -e .
```

## Correctness Gate

Always run before any benchmarking:
```bash
python tests/test_correctness.py
```
All ops must pass `np.allclose(ts_out, numpy_out, atol=1e-5)`.

## Benchmark Suite

```bash
python benchmarks/bench_elementwise.py   # primary claim
python benchmarks/bench_mlp.py           # secondary claim
python benchmarks/bench_sweep.py         # sweep array sizes + chain lengths
python benchmarks/bench_memory.py        # heap usage
python benchmarks/plot_results.py        # generate PNG charts
```

## LLVM Version

Targets LLVM 17. Key APIs used:
- `llvm::orc::LLJITBuilder` (OrcJIT)
- `llvm::PassBuilder` with `OptimizationLevel::O3`
- Loop vectorize metadata: `!llvm.loop.vectorize.enable`
- Intrinsics: `llvm.maxnum.f32`, `llvm.exp.f32`, `llvm.log.f32`, `llvm.sqrt.f32`

## File Map

| Path | Role |
|---|---|
| `include/ts/ir.h` | Node, Graph, TensorType, OpKind, FusedOp |
| `include/ts/ops.h` | OpInfo: is_elementwise, is_binary, llvm_intrinsic |
| `include/ts/passes.h` | PassBase + all pass declarations |
| `include/ts/codegen.h` | LLVMCodegen |
| `include/ts/jit.h` | OrcJIT wrapper |
| `include/ts/runtime.h` | BufferPool, CompiledFunction |
| `src/passes/fusion.cpp` | Greedy element-wise chain fusion |
| `src/passes/matmul_epilogue.cpp` | Matmul+BiasAdd+activation → FusedMatmul |
| `src/passes/buffer_reuse.cpp` | Liveness-based buffer slot assignment |
| `src/codegen.cpp` | LLVM IR generation for all kernel types |
| `src/jit.cpp` | OrcJIT module add + symbol lookup |
| `src/runtime.cpp` | BufferPool allocation + CompiledFunction::operator() |
| `bindings/python.cpp` | pybind11 module `tensorscript` |
| `benchmarks/` | All benchmark scripts |
| `tests/test_correctness.py` | Numerical correctness gate |
| `results/` | Committed benchmark artifacts |
