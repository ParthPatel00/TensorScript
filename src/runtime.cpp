#include "ts/runtime.h"
#include "ts/ir.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

// BLAS headers — on macOS use Accelerate, on Linux use OpenBLAS/cblas
#ifdef __APPLE__
#  define ACCELERATE_NEW_LAPACK  // suppresses cblas deprecation warnings on macOS 13.3+
#  include <Accelerate/Accelerate.h>
#else
#  include <cblas.h>
#endif

namespace ts {

void BufferPool::allocate(const Graph& g) {
    // Find max slot index and required capacity per slot
    std::unordered_map<int, int64_t> slot_cap;
    auto order = g.topo_order();
    for (auto* n : order) {
        if (n->buffer_slot < 0) continue;
        int64_t sz = n->output_type.numel();
        auto it = slot_cap.find(n->buffer_slot);
        if (it == slot_cap.end() || it->second < sz)
            slot_cap[n->buffer_slot] = sz;
    }
    int num = (int)slot_cap.size();
    if (num == 0) return;
    int max_slot = 0;
    for (auto& [s, _] : slot_cap) max_slot = std::max(max_slot, s);
    slots.resize(max_slot + 1);
    for (auto& [s, cap] : slot_cap) slots[s].resize(cap, 0.f);
}

// Special slot encoding: negative means graph input. Slot -1 = input 0, -2 = input 1, etc.
static constexpr int graph_input_slot(int i) { return -(i + 1); }
static bool is_graph_input_slot(int s) { return s < 0; }
static int input_index(int s) { return -(s + 1); }

// EpilogueFn is declared in runtime.h; redeclare locally for clarity.
static_assert(sizeof(EpilogueFn) == sizeof(KernelFn),
              "function pointer sizes must match for safe reinterpret_cast");

CompiledFunction build_runtime(const Graph& g,
                                const std::vector<std::pair<std::string, Node*>>& kernels,
                                TensorScriptJIT& jit) {
    CompiledFunction cf;
    cf.num_graph_inputs = (int)g.inputs.size();
    cf.pool.allocate(g);

    // Map node* -> its output slot
    std::unordered_map<Node*, int> node_slot;
    auto order = g.topo_order();
    for (int i = 0; i < (int)g.inputs.size(); i++) {
        node_slot[g.inputs[i]] = graph_input_slot(i);
    }
    for (auto* n : order) {
        if (n->buffer_slot >= 0) node_slot[n] = n->buffer_slot;
    }

    cf.output_slot = node_slot.count(g.output) ? node_slot[g.output] : -1;
    cf.numel = g.output ? g.output->output_type.numel() : 0;

    for (auto& [kname, node] : kernels) {
        KernelCall kc;
        kc.numel = node->output_type.numel();
        kc.output_slot = node->buffer_slot;

        if (node->kind == OpKind::FusedKernel) {
            kc.fn = jit.lookup(kname);
            for (auto* inp : node->inputs) {
                assert(node_slot.count(inp));
                kc.input_slots.push_back(node_slot[inp]);
            }
        } else if (node->kind == OpKind::FusedMatmul) {
            // Cast epilogue fn; BLAS dispatch happens in execute()
            kc.epilogue_fn = reinterpret_cast<EpilogueFn>(
                reinterpret_cast<void(*)()>(jit.lookup(kname)));
            // inputs: [A, B] or [A, B, bias_vec]
            kc.has_bias = (node->inputs.size() >= 3);
            for (auto* inp : node->inputs) {
                assert(node_slot.count(inp));
                kc.input_slots.push_back(node_slot[inp]);
            }
            // Extract GEMM dims: A=[M,K], B=[K,N]
            auto& a_shape = node->inputs[0]->output_type.shape;
            auto& b_shape = node->inputs[1]->output_type.shape;
            kc.M = (a_shape.size() >= 2) ? a_shape[0] : 1;
            kc.K = a_shape.back();
            kc.N = b_shape.back();
        }
        cf.calls.push_back(std::move(kc));
    }

    return cf;
}

float* CompiledFunction::execute(std::vector<float*> user_inputs) {
    if ((int)user_inputs.size() != num_graph_inputs) {
        throw std::runtime_error("execute: wrong number of inputs");
    }

    auto resolve = [&](int slot) -> float* {
        if (is_graph_input_slot(slot))
            return user_inputs[input_index(slot)];
        return pool.get(slot);
    };

    for (auto& kc : calls) {
        float* out = pool.get(kc.output_slot);

        std::vector<float*> in_ptrs;
        for (int s : kc.input_slots) in_ptrs.push_back(resolve(s));

        if (kc.epilogue_fn) {
            // FusedMatmul: BLAS GEMM then fused epilogue (bias + activation) in-place.
            float* A    = in_ptrs[0];
            float* B    = in_ptrs[1];
            // When there is no bias, pass 'out' as a harmless readable dummy pointer.
            // The epilogue selects v (not v+bias) when has_bias=false, so no out-of-bounds.
            float* bias = kc.has_bias ? in_ptrs[2] : out;

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)kc.M, (int)kc.N, (int)kc.K,
                        1.0f, A, (int)kc.K,
                              B, (int)kc.N,
                        0.0f, out, (int)kc.N);

            // Epilogue runs in-place: reads mm_out, writes final_out (same buffer).
            kc.epilogue_fn(out, bias, out, kc.numel, kc.has_bias);
        } else if (kc.fn) {
            // FusedKernel: void kernel(float** inputs, int num, float* out, i64 n)
            kc.fn(in_ptrs.data(), (int)in_ptrs.size(), out, kc.numel);
        }
    }

    return pool.get(output_slot);
}

} // namespace ts
