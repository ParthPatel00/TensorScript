#include "ts/passes.h"
#include <iostream>

namespace ts {

// Ops that have a direct Apple vForce equivalent or can be efficiently composed
// from vForce primitives (sigmoid = vneg + vvexpf + vsadd + svdiv).
static bool is_transcendental(OpKind k) {
    return k == OpKind::Sigmoid
        || k == OpKind::Tanh
        || k == OpKind::Exp
        || k == OpKind::Log;
}

static VecLibFn to_veclib_fn(OpKind k) {
    switch (k) {
    case OpKind::Sigmoid: return VecLibFn::Sigmoid;
    case OpKind::Tanh:    return VecLibFn::Tanh;
    case OpKind::Exp:     return VecLibFn::Exp;
    case OpKind::Log:     return VecLibFn::Log;
    default:
        return VecLibFn::Exp; // unreachable
    }
}

void VecLibSplitPass::run(Graph& g) {
#ifdef __APPLE__
    int converted = 0;
    for (auto& node_ptr : g.nodes) {
        auto* n = node_ptr.get();
        if (n->is_dead) continue;
        if (!is_transcendental(n->kind)) continue;

        // Only convert unary transcendentals (all of these are unary).
        // The node's single input stays; we just change the dispatch kind.
        n->veclib_fn = to_veclib_fn(n->kind);
        n->kind = OpKind::VecLibCall;
        converted++;
    }
    if (converted > 0) {
        std::cout << "[Fuse] VecLibSplitPass: converted "
                  << converted << " transcendental op(s) to Apple vForce dispatch\n";
    }
#else
    (void)g; // no-op on non-Apple platforms
#endif
}

} // namespace ts
