#include "ts/passes.h"
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace ts {

// Count how many live nodes consume node n.
static int consumer_count(Node* n, const std::vector<Node*>& order) {
    int cnt = 0;
    for (auto* node : order) {
        if (node->is_dead) continue;
        for (auto* inp : node->inputs) {
            if (inp == n) { cnt++; break; }
        }
    }
    return cnt;
}

// Merge node src (element-wise) into its sole consumer dst (which becomes / is a FusedKernel).
// Returns true if merge happened.
static bool try_merge(Node* src, Node* dst, const std::vector<Node*>& order) {
    if (!is_elementwise(src->kind) && src->kind != OpKind::FusedKernel) return false;
    if (!is_elementwise(dst->kind) && dst->kind != OpKind::FusedKernel) return false;

    // src must have exactly one consumer and that consumer must be dst
    if (consumer_count(src, order) != 1) return false;
    bool dst_consumes_src = false;
    for (auto* inp : dst->inputs) if (inp == src) { dst_consumes_src = true; break; }
    if (!dst_consumes_src) return false;

    // Build the new FusedKernel in dst.
    // Collect src's fused_ops (or the single op if src is not yet fused).
    std::vector<FusedOp> src_ops;
    std::vector<Node*> src_inputs;

    if (src->kind == OpKind::FusedKernel) {
        src_ops   = src->fused_ops;
        src_inputs = src->inputs;
    } else {
        // src is a plain element-wise op
        src_inputs = src->inputs;  // the raw graph inputs feeding this op
        if (is_binary(src->kind)) {
            // src has two inputs: [left, right]
            // acc will be initialized to left; the op uses right (index 1 in src_inputs)
            src_ops.push_back({src->kind, 1});
        } else {
            src_ops.push_back({src->kind, -1});
        }
    }

    // Build new input list for the merged kernel:
    // Start with src_inputs, then add any of dst's inputs that aren't src itself.
    std::vector<Node*> new_inputs = src_inputs;

    // Map from node ptr -> index in new_inputs
    std::unordered_map<Node*, int> input_map;
    for (int i = 0; i < (int)new_inputs.size(); i++) {
        input_map[new_inputs[i]] = i;
    }

    // Now figure out dst's ops. If dst is already a FusedKernel, its ops reference
    // indices into dst->inputs. We need to remap those indices into new_inputs.
    // dst's inputs are: [src, possibly other nodes]
    std::vector<FusedOp> dst_ops;

    if (dst->kind == OpKind::FusedKernel) {
        // Remap dst's existing fused_ops
        // Build mapping: old dst input idx -> new input idx
        std::vector<int> remap(dst->inputs.size(), -1);
        for (int i = 0; i < (int)dst->inputs.size(); i++) {
            Node* di = dst->inputs[i];
            if (di == src) {
                remap[i] = -2;  // sentinel: this was the accumulated value from src
            } else {
                if (!input_map.count(di)) {
                    input_map[di] = (int)new_inputs.size();
                    new_inputs.push_back(di);
                }
                remap[i] = input_map[di];
            }
        }
        for (auto& op : dst->fused_ops) {
            if (op.input_idx == -1) {
                dst_ops.push_back(op);  // unary: unchanged
            } else {
                int new_idx = remap[op.input_idx];
                dst_ops.push_back({op.kind, new_idx == -2 ? -1 : new_idx});
            }
        }
    } else {
        // dst is a plain element-wise op; convert it
        if (is_binary(dst->kind)) {
            // dst->inputs = [src, other] or [other, src]
            // Find which input is src and which is the extra
            for (int i = 0; i < (int)dst->inputs.size(); i++) {
                if (dst->inputs[i] != src) {
                    // This is the extra input
                    if (!input_map.count(dst->inputs[i])) {
                        input_map[dst->inputs[i]] = (int)new_inputs.size();
                        new_inputs.push_back(dst->inputs[i]);
                    }
                    dst_ops.push_back({dst->kind, input_map[dst->inputs[i]]});
                    goto done_dst;
                }
            }
            // Both inputs are src (e.g., x*x), use accumulated value twice
            dst_ops.push_back({dst->kind, -1});
            done_dst:;
        } else {
            dst_ops.push_back({dst->kind, -1});
        }
    }

    // Commit: turn dst into a FusedKernel
    dst->kind = OpKind::FusedKernel;
    dst->fused_ops = src_ops;
    for (auto& op : dst_ops) dst->fused_ops.push_back(op);
    dst->inputs = std::move(new_inputs);
    src->is_dead = true;

    return true;
}

void FusionPass::run(Graph& g) {
    int total_merged = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        auto order = g.topo_order();  // recompute each pass
        for (auto* n : order) {
            if (n->is_dead) continue;
            if (!is_elementwise(n->kind) && n->kind != OpKind::FusedKernel) continue;
            // Find this node's consumers
            for (auto* candidate : order) {
                if (candidate->is_dead || candidate == n) continue;
                if (try_merge(n, candidate, order)) {
                    changed = true;
                    total_merged++;
                    break;
                }
            }
            if (changed) break;  // restart after each merge
        }
    }
    if (total_merged > 0) {
        std::cout << "[TensorScript] FusionPass: merged " << total_merged << " node(s) into FusedKernel(s)\n";
    }

    // Wrap any remaining live elementwise nodes that weren't merged into a chain.
    // These are isolated ops (e.g. a single Add that is also the output) — codegen
    // only emits functions for FusedKernel, so we must convert them here.
    int wrapped = 0;
    for (auto* n : g.topo_order()) {
        if (n->is_dead || !is_elementwise(n->kind)) continue;
        n->fused_ops = {FusedOp{n->kind, is_binary(n->kind) ? 1 : -1}};
        n->kind = OpKind::FusedKernel;
        wrapped++;
    }
    if (wrapped > 0) {
        std::cout << "[TensorScript] FusionPass: wrapped " << wrapped
                  << " isolated elementwise node(s) into FusedKernel(s)\n";
    }
}

} // namespace ts
