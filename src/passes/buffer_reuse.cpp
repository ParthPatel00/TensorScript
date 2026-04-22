#include "ts/passes.h"
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace ts {

// Liveness-based buffer slot assignment.
// Each kernel node's output needs a buffer.
// We track live ranges [birth, death] in topological index.
// Nodes that are inputs (user-provided) don't need a pool slot.
// A slot is reused if its previous occupant's live range ended before the new one begins.

struct LiveRange {
    int birth;  // topological index where value is produced
    int death;  // last topological index where value is consumed
    Node* node;
};

void BufferReusePass::run(Graph& g) {
    auto order = g.topo_order();
    int n = (int)order.size();

    // Map node* -> topo index
    std::unordered_map<Node*, int> idx;
    for (int i = 0; i < n; i++) idx[order[i]] = i;

    // Compute death = max consumer index
    std::unordered_map<Node*, int> death;
    for (auto* node : order) {
        if (node->is_dead) continue;
        for (auto* inp : node->inputs) {
            if (!death.count(inp) || death[inp] < idx[node])
                death[inp] = idx[node];
        }
    }

    // Collect live ranges for nodes that need output buffers (non-input, non-constant, non-dead)
    std::vector<LiveRange> ranges;
    for (int i = 0; i < n; i++) {
        auto* node = order[i];
        if (node->is_dead) continue;
        if (node->kind == OpKind::Input || node->kind == OpKind::Constant) continue;
        int d = death.count(node) ? death[node] : i;
        ranges.push_back({i, d, node});
    }

    // Greedy slot assignment: scan ranges in birth order, reuse a free slot if size matches or slot can hold it.
    // We use a simple approach: slot is free once its current occupant's range ends.
    struct Slot {
        int free_after;  // topo index after which slot is free again
        int64_t capacity; // max numel allocated for this slot
    };
    std::vector<Slot> slots;

    for (auto& lr : ranges) {
        int64_t need = lr.node->output_type.numel();
        // Find a free slot with sufficient capacity
        int best = -1;
        for (int s = 0; s < (int)slots.size(); s++) {
            if (slots[s].free_after < lr.birth && slots[s].capacity >= need) {
                best = s; break;
            }
        }
        if (best == -1) {
            // Allocate new slot
            best = (int)slots.size();
            slots.push_back({lr.death, need});
        } else {
            slots[best].free_after = lr.death;
            slots[best].capacity = std::max(slots[best].capacity, need);
        }
        lr.node->buffer_slot = best;
    }

    num_slots = (int)slots.size();
    std::cout << "[Fuse] BufferReusePass: "
              << ranges.size() << " buffers → " << num_slots << " slot(s)\n";
}

// BufferPool implementation lives in runtime.cpp to avoid circular deps.

} // namespace ts
