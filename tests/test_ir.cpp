#include <gtest/gtest.h>
#include "ts/ir.h"

using namespace ts;

TEST(IR, BuildSimpleGraph) {
    Graph g;
    TensorType t; t.shape = {1000000};
    auto* a = g.add_input("a", t);
    auto* b = g.add_input("b", t);
    auto* x = g.add_op(OpKind::Add, {a, b});
    auto* y = g.add_op(OpKind::Relu, {x});
    g.set_output(y);

    EXPECT_EQ(g.inputs.size(), 2u);
    auto order = g.topo_order();
    EXPECT_EQ(order.size(), 4u);  // a, b, Add, Relu
}

TEST(IR, TopoOrder) {
    Graph g;
    TensorType t; t.shape = {100};
    auto* a = g.add_input("a", t);
    auto* b = g.add_input("b", t);
    auto* c = g.add_op(OpKind::Add, {a, b});
    auto* d = g.add_op(OpKind::Sigmoid, {c});
    g.set_output(d);

    auto order = g.topo_order();
    // a and b must come before c, c before d
    auto pos = [&](Node* n) {
        for (int i = 0; i < (int)order.size(); i++) if (order[i] == n) return i;
        return -1;
    };
    EXPECT_LT(pos(a), pos(c));
    EXPECT_LT(pos(b), pos(c));
    EXPECT_LT(pos(c), pos(d));
}

TEST(IR, IsElementwise) {
    EXPECT_TRUE(is_elementwise(OpKind::Add));
    EXPECT_TRUE(is_elementwise(OpKind::Relu));
    EXPECT_TRUE(is_elementwise(OpKind::Sigmoid));
    EXPECT_FALSE(is_elementwise(OpKind::Matmul));
    EXPECT_FALSE(is_elementwise(OpKind::Input));
}

TEST(IR, DotExport) {
    Graph g;
    TensorType t; t.shape = {100};
    auto* a = g.add_input("a", t);
    auto* b = g.add_op(OpKind::Relu, {a});
    g.set_output(b);
    // Should not throw
    g.to_dot("/tmp/test_graph.dot");
}
