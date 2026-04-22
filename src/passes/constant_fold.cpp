#include "ts/passes.h"
#include <cmath>

namespace ts {

void ConstantFoldPass::run(Graph& g) {
    // If all inputs of a non-input, non-output node are Constants,
    // evaluate it at compile time and replace with a Constant node.
    auto order = g.topo_order();
    for (auto* n : order) {
        if (n->kind == OpKind::Constant || n->kind == OpKind::Input) continue;
        if (!is_elementwise(n->kind)) continue;
        bool all_const = !n->inputs.empty();
        for (auto* inp : n->inputs) {
            if (inp->kind != OpKind::Constant) { all_const = false; break; }
        }
        if (!all_const) continue;

        int64_t sz = n->output_type.numel();
        std::vector<float> out(sz);
        auto& a = n->inputs[0]->const_data;
        std::vector<float> b_empty;
        auto& b = (n->inputs.size() > 1) ? n->inputs[1]->const_data : b_empty;

        for (int64_t i = 0; i < sz; i++) {
            float av = a[i];
            float bv = b.empty() ? 0.f : b[i];
            switch (n->kind) {
            case OpKind::Add:     out[i] = av + bv; break;
            case OpKind::Sub:     out[i] = av - bv; break;
            case OpKind::Mul:     out[i] = av * bv; break;
            case OpKind::Div:     out[i] = av / bv; break;
            case OpKind::Relu:    out[i] = av > 0 ? av : 0; break;
            case OpKind::Sigmoid: out[i] = 1.f / (1.f + std::exp(-av)); break;
            case OpKind::Tanh:    out[i] = std::tanh(av); break;
            case OpKind::Exp:     out[i] = std::exp(av); break;
            case OpKind::Log:     out[i] = std::log(av); break;
            case OpKind::Neg:     out[i] = -av; break;
            case OpKind::Sqrt:    out[i] = std::sqrt(av); break;
            default: break;
            }
        }
        n->kind = OpKind::Constant;
        n->const_data = std::move(out);
        n->inputs.clear();
    }
}

} // namespace ts
