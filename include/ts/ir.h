#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ts {

enum class DataType { Float32, Float16 };

struct TensorType {
    std::vector<int64_t> shape;
    DataType dtype = DataType::Float32;

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    int64_t nbytes() const {
        return numel() * (dtype == DataType::Float32 ? 4 : 2);
    }
};

enum class OpKind {
    Input,
    Constant,
    Add, Sub, Mul, Div,
    Relu, Sigmoid, Tanh, Exp, Log, Neg, Sqrt,
    Matmul,
    BiasAdd,
    FusedKernel,  // element-wise chain collapsed by FusionPass
    FusedMatmul,  // Matmul + optional BiasAdd + optional activation
};

// One step inside a FusedKernel loop body.
// acc starts as inputs[0][i]; each step updates acc.
// input_idx == -1: unary  →  acc = op(acc)
// input_idx >= 0 : binary →  acc = acc OP inputs[input_idx][i]
struct FusedOp {
    OpKind kind;
    int input_idx = -1;
};

struct Node {
    std::string name;
    OpKind kind;
    TensorType output_type;
    std::vector<Node*> inputs;

    // FusedKernel
    std::vector<FusedOp> fused_ops;

    // FusedMatmul: post-matmul activation; OpKind::Input means none
    OpKind epilogue = OpKind::Input;

    // Assigned by BufferReusePass; -1 = not yet assigned
    int buffer_slot = -1;

    // Constant data (only for OpKind::Constant)
    std::vector<float> const_data;

    bool is_dead = false;
};

struct Graph {
    std::vector<std::unique_ptr<Node>> nodes;
    std::vector<Node*> inputs;
    Node* output = nullptr;

    Node* add_input(const std::string& name, TensorType type);
    Node* add_op(OpKind kind, std::vector<Node*> ins, const std::string& name = "");
    Node* add_const(const std::string& name, TensorType type, std::vector<float> data);
    void  set_output(Node* n);
    void  dump() const;
    void  to_dot(const std::string& path) const;

    // Topological order (inputs first, output last)
    std::vector<Node*> topo_order() const;
};

bool is_elementwise(OpKind k);
bool is_binary(OpKind k);
const char* op_name(OpKind k);

} // namespace ts
