#include "ts/passes.h"
#include <iostream>

namespace ts {

// Pattern: Matmul -> BiasAdd -> [Relu | Sigmoid | Tanh]  (all single-consumer)
// Collapses to FusedMatmul with epilogue annotation.

static int consumer_count_ep(Node* n, const std::vector<Node*>& order) {
    int cnt = 0;
    for (auto* node : order) {
        if (node->is_dead) continue;
        for (auto* inp : node->inputs) if (inp == n) { cnt++; break; }
    }
    return cnt;
}

void MatmulEpiloguePass::run(Graph& g) {
    auto order = g.topo_order();
    int fused = 0;

    for (auto* mm : order) {
        if (mm->is_dead || mm->kind != OpKind::Matmul) continue;
        if (consumer_count_ep(mm, order) != 1) continue;

        // Find the sole consumer
        Node* bias = nullptr;
        for (auto* n : order) {
            if (n->is_dead) continue;
            for (auto* inp : n->inputs) {
                if (inp == mm && n->kind == OpKind::BiasAdd) {
                    bias = n; break;
                }
            }
            if (bias) break;
        }

        OpKind activation = OpKind::Input;  // none
        Node* act = nullptr;

        if (bias && consumer_count_ep(bias, order) == 1) {
            // Check for activation after bias
            for (auto* n : order) {
                if (n->is_dead) continue;
                for (auto* inp : n->inputs) {
                    if (inp == bias) {
                        OpKind k = n->kind;
                        if (k == OpKind::Relu || k == OpKind::Sigmoid || k == OpKind::Tanh) {
                            act = n; activation = k;
                        }
                        break;
                    }
                }
                if (act) break;
            }
        }

        // Merge into FusedMatmul on mm node
        Node* anchor = act ? act : (bias ? bias : nullptr);
        if (!anchor) continue;  // nothing to fuse beyond the matmul

        mm->kind = OpKind::FusedMatmul;
        mm->epilogue = activation;

        if (bias) {
            // Add bias vector as extra input
            // bias->inputs = [mm, bias_vec]
            for (auto* inp : bias->inputs) {
                if (inp != mm) mm->inputs.push_back(inp);
            }
            bias->is_dead = true;
        }
        if (act) {
            act->is_dead = true;
            // Rewire anything that consumed act to consume mm
            for (auto* n : order) {
                for (auto*& inp : n->inputs) {
                    if (inp == act) inp = mm;
                }
            }
            if (g.output == act) g.output = mm;
        } else if (bias) {
            for (auto* n : order) {
                for (auto*& inp : n->inputs) {
                    if (inp == bias) inp = mm;
                }
            }
            if (g.output == bias) g.output = mm;
        }

        fused++;
    }

    if (fused > 0)
        std::cout << "[Fuse] MatmulEpiloguePass: fused " << fused << " Matmul(s)\n";
}

} // namespace ts
