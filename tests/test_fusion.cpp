#include <gtest/gtest.h>
#include "ts/ir.h"
#include "ts/passes.h"

using namespace ts;

TEST(Fusion, BasicChain) {
    Graph g;
    TensorType t; t.shape = {1000};
    auto* a = g.add_input("a", t);
    auto* b = g.add_input("b", t);
    auto* x0 = g.add_op(OpKind::Add,  {a, b});
    auto* x1 = g.add_op(OpKind::Relu, {x0});
    auto* x2 = g.add_op(OpKind::Mul,  {x1, b});
    g.set_output(x2);

    // Before fusion: 5 nodes (a, b, Add, Relu, Mul)
    EXPECT_EQ(g.topo_order().size(), 5u);

    FusionPass{}.run(g);

    // After fusion: a, b, and one FusedKernel
    auto order = g.topo_order();
    // Count non-dead nodes
    int live = 0; Node* fused = nullptr;
    for (auto& n : g.nodes) {
        if (!n->is_dead) {
            live++;
            if (n->kind == OpKind::FusedKernel) fused = n.get();
        }
    }
    EXPECT_EQ(live, 3);  // a, b, FusedKernel
    ASSERT_NE(fused, nullptr);
    // FusedKernel should have 3 ops: Add, Relu, Mul
    EXPECT_EQ(fused->fused_ops.size(), 3u);
    EXPECT_EQ(fused->fused_ops[0].kind, OpKind::Add);
    EXPECT_EQ(fused->fused_ops[1].kind, OpKind::Relu);
    EXPECT_EQ(fused->fused_ops[2].kind, OpKind::Mul);
}

TEST(Fusion, UnaryChain) {
    Graph g;
    TensorType t; t.shape = {100};
    auto* a = g.add_input("a", t);
    auto* x0 = g.add_op(OpKind::Relu,    {a});
    auto* x1 = g.add_op(OpKind::Sigmoid, {x0});
    auto* x2 = g.add_op(OpKind::Tanh,    {x1});
    g.set_output(x2);

    FusionPass{}.run(g);

    Node* fused = nullptr;
    for (auto& n : g.nodes) {
        if (!n->is_dead && n->kind == OpKind::FusedKernel) fused = n.get();
    }
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->fused_ops.size(), 3u);
}

TEST(BufferReuse, PingPong) {
    Graph g;
    TensorType t; t.shape = {1000};
    auto* a = g.add_input("a", t);
    auto* b = g.add_input("b", t);
    auto* x0 = g.add_op(OpKind::Add,     {a, b});
    auto* x1 = g.add_op(OpKind::Relu,    {x0});
    auto* x2 = g.add_op(OpKind::Mul,     {x1, b});
    auto* x3 = g.add_op(OpKind::Sigmoid, {x2});
    auto* x4 = g.add_op(OpKind::Tanh,    {x3});
    g.set_output(x4);

    FusionPass{}.run(g);
    DCEPass{}.run(g);
    BufferReusePass br;
    br.run(g);

    // A single FusedKernel only needs 1 output slot
    EXPECT_LE(br.num_slots, 2);
}
