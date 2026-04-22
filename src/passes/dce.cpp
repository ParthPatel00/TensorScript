#include "ts/passes.h"
#include <unordered_set>

namespace ts {

void DCEPass::run(Graph& g) {
    // Mark all nodes reachable from output as live.
    std::unordered_set<Node*> live;
    std::function<void(Node*)> mark = [&](Node* n) {
        if (!n || live.count(n)) return;
        live.insert(n);
        for (auto* inp : n->inputs) mark(inp);
    };
    mark(g.output);

    for (auto& n : g.nodes) {
        if (!live.count(n.get())) n->is_dead = true;
    }
}

} // namespace ts
