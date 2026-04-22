#pragma once
#include "ts/ir.h"
#include "ts/jit.h"
#include <vector>
#include <memory>

namespace ts {

// Matmul epilogue function: (matmul_out, bias_or_dummy, final_out, n, has_bias)
// Emitted by codegen for FusedMatmul nodes; runs after BLAS fills matmul_out.
using EpilogueFn = void(*)(float*, float*, float*, int64_t, bool);

// Pre-allocated slab of float buffers indexed by slot number.
// BufferReusePass assigns buffer_slot to each Node; the pool
// allocates one allocation per unique slot.
struct BufferPool {
    std::vector<std::vector<float>> slots;

    void allocate(const Graph& g);
    float* get(int slot) { return slots[slot].data(); }
};

// A compiled, ready-to-call Fuse function.
struct KernelCall {
    KernelFn   fn          = nullptr;  // FusedKernel: void(float**, int, float*, int64_t)
    EpilogueFn epilogue_fn = nullptr;  // FusedMatmul: void(float*, float*, float*, int64_t, bool)
    std::vector<int> input_slots;      // pool slot or negative graph-input sentinel
    int     output_slot = -1;
    int64_t numel       = 0;
    // GEMM dimensions (only when epilogue_fn != nullptr)
    int64_t M = 0, K = 0, N = 0;
    bool    has_bias = false;
    // VecLibCall dispatch (Apple vecLib; ignored on Linux where is_veclib is always false)
    bool      is_veclib  = false;
    VecLibFn  veclib_fn  = VecLibFn::Exp;
};

// Special sentinel: graph input i is stored at slot -(i+1) (negative)
// The executor resolves these to actual user-provided float* pointers.

struct CompiledFunction {
    std::vector<KernelCall> calls;
    BufferPool pool;
    int num_graph_inputs = 0;
    int output_slot = -1;
    int64_t numel = 0;

    // Accepts raw float* arrays (one per graph input), returns raw float* output.
    // The returned pointer is into pool's output slot — valid until next call.
    float* execute(std::vector<float*> inputs);
};

// Build a CompiledFunction from a compiled graph + JIT.
CompiledFunction build_runtime(const Graph& g,
                                const std::vector<std::pair<std::string, Node*>>& kernels,
                                FuseJIT& jit);

} // namespace ts
