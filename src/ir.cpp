#include "ts/ir.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace ts {

bool is_elementwise(OpKind k) {
    switch (k) {
    case OpKind::Add: case OpKind::Sub: case OpKind::Mul: case OpKind::Div:
    case OpKind::Relu: case OpKind::Sigmoid: case OpKind::Tanh:
    case OpKind::Exp: case OpKind::Log: case OpKind::Neg: case OpKind::Sqrt:
        return true;
    default:
        return false;
    }
}

bool is_binary(OpKind k) {
    switch (k) {
    case OpKind::Add: case OpKind::Sub: case OpKind::Mul: case OpKind::Div:
        return true;
    default:
        return false;
    }
}

const char* op_name(OpKind k) {
    switch (k) {
    case OpKind::Input:       return "Input";
    case OpKind::Constant:    return "Constant";
    case OpKind::Add:         return "Add";
    case OpKind::Sub:         return "Sub";
    case OpKind::Mul:         return "Mul";
    case OpKind::Div:         return "Div";
    case OpKind::Relu:        return "Relu";
    case OpKind::Sigmoid:     return "Sigmoid";
    case OpKind::Tanh:        return "Tanh";
    case OpKind::Exp:         return "Exp";
    case OpKind::Log:         return "Log";
    case OpKind::Neg:         return "Neg";
    case OpKind::Sqrt:        return "Sqrt";
    case OpKind::Matmul:      return "Matmul";
    case OpKind::BiasAdd:     return "BiasAdd";
    case OpKind::FusedKernel: return "FusedKernel";
    case OpKind::FusedMatmul: return "FusedMatmul";
    case OpKind::VecLibCall:  return "VecLibCall";
    default:                  return "Unknown";
    }
}

Node* Graph::add_input(const std::string& name, TensorType type) {
    auto n = std::make_unique<Node>();
    n->name = name;
    n->kind = OpKind::Input;
    n->output_type = std::move(type);
    Node* ptr = n.get();
    nodes.push_back(std::move(n));
    inputs.push_back(ptr);
    return ptr;
}

Node* Graph::add_op(OpKind kind, std::vector<Node*> ins, const std::string& name) {
    auto n = std::make_unique<Node>();
    n->name = name.empty() ? op_name(kind) : name;
    n->kind = kind;
    n->inputs = std::move(ins);
    // Infer output type
    if (!n->inputs.empty()) {
        if (kind == OpKind::Matmul && n->inputs.size() == 2) {
            // [M, K] x [K, N] -> [M, N]; [K] x [K, N] -> [N]
            auto& a = n->inputs[0]->output_type;
            auto& b = n->inputs[1]->output_type;
            TensorType out; out.dtype = a.dtype;
            if (a.shape.size() >= 2 && b.shape.size() >= 2) {
                out.shape = {a.shape[0], b.shape.back()};
            } else if (a.shape.size() == 1 && b.shape.size() == 2) {
                out.shape = {b.shape.back()};
            } else {
                out.shape = a.shape;
            }
            n->output_type = out;
        } else {
            n->output_type = n->inputs[0]->output_type;
        }
    }
    Node* ptr = n.get();
    nodes.push_back(std::move(n));
    return ptr;
}

Node* Graph::add_const(const std::string& name, TensorType type, std::vector<float> data) {
    auto n = std::make_unique<Node>();
    n->name = name;
    n->kind = OpKind::Constant;
    n->output_type = std::move(type);
    n->const_data = std::move(data);
    Node* ptr = n.get();
    nodes.push_back(std::move(n));
    return ptr;
}

void Graph::set_output(Node* n) { output = n; }

std::vector<Node*> Graph::topo_order() const {
    std::vector<Node*> order;
    std::unordered_set<Node*> visited;

    std::function<void(Node*)> visit = [&](Node* n) {
        if (!n || visited.count(n)) return;
        visited.insert(n);
        for (auto* inp : n->inputs) visit(inp);
        order.push_back(n);
    };

    visit(output);
    return order;
}

void Graph::dump() const {
    auto order = topo_order();
    std::cout << "Graph (" << order.size() << " nodes):\n";
    for (auto* n : order) {
        std::cout << "  [" << op_name(n->kind) << "] " << n->name
                  << " shape=[";
        for (size_t i = 0; i < n->output_type.shape.size(); i++) {
            if (i) std::cout << ",";
            std::cout << n->output_type.shape[i];
        }
        std::cout << "]";
        if (n->kind == OpKind::FusedKernel) {
            std::cout << " fused=[";
            for (size_t i = 0; i < n->fused_ops.size(); i++) {
                if (i) std::cout << ",";
                std::cout << op_name(n->fused_ops[i].kind);
            }
            std::cout << "] inputs=" << n->inputs.size();
        }
        std::cout << "\n";
    }
}

void Graph::to_dot(const std::string& path) const {
    std::ofstream f(path);
    f << "digraph G {\n  rankdir=TB;\n  node [shape=box];\n";
    auto order = topo_order();
    for (auto* n : order) {
        std::string label = op_name(n->kind);
        label += "\\n" + n->name;
        if (n->kind == OpKind::FusedKernel) {
            label += "\\n[";
            for (size_t i = 0; i < n->fused_ops.size(); i++) {
                if (i) label += ",";
                label += op_name(n->fused_ops[i].kind);
            }
            label += "]";
        }
        f << "  \"" << (void*)n << "\" [label=\"" << label << "\"];\n";
        for (auto* inp : n->inputs) {
            f << "  \"" << (void*)inp << "\" -> \"" << (void*)n << "\";\n";
        }
    }
    f << "}\n";
}

} // namespace ts
