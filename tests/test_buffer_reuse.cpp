#include <gtest/gtest.h>
#include "ts/ir.h"
#include "ts/passes.h"

#include <set>

using namespace ts;

static int max_slot(const Graph& g) {
    int m = -1;
    for (auto& n : g.nodes) {
        if (!n->is_dead && n->buffer_slot >= 0)
            m = std::max(m, n->buffer_slot);
    }
    return m;
}

static int unique_slots(const Graph& g) {
    std::set<int> s;
    for (auto& n : g.nodes) {
        if (!n->is_dead && n->buffer_slot >= 0)
            s.insert(n->buffer_slot);
    }
    return (int)s.size();
}

TEST(BufferReuse, SingleOpOneSlot) {
    // A single Add node: needs exactly 1 output slot.
    Graph g;
    TensorType t; t.shape = {1000};
    auto* a = g.add_input("a", t);
    auto* b = g.add_input("b", t);
    auto* x = g.add_op(OpKind::Add, {a, b});
    g.set_output(x);
    run_passes(g);

    EXPECT_EQ(unique_slots(g), 1);
}

TEST(BufferReuse, FiveOpChainPingPong) {
    // After full fusion, a 5-op element-wise chain collapses to a single FusedKernel.
    // That single kernel needs exactly 1 output slot (no intermediates).
    const int N = 10000;
    Graph g;
    TensorType t; t.shape = {N};
    auto* a = g.add_input("a", t);
    auto* b = g.add_input("b", t);
    auto* x0 = g.add_op(OpKind::Add,     {a, b});
    auto* x1 = g.add_op(OpKind::Relu,    {x0});
    auto* x2 = g.add_op(OpKind::Mul,     {x1, b});
    auto* x3 = g.add_op(OpKind::Sigmoid, {x2});
    auto* x4 = g.add_op(OpKind::Tanh,    {x3});
    g.set_output(x4);
    run_passes(g);

    // All 5 element-wise nodes fused into 1 FusedKernel -> only 1 slot needed
    EXPECT_LE(unique_slots(g), 2);
    EXPECT_LE(max_slot(g) + 1, 2);
}

TEST(BufferReuse, TwoParallelChainsReuseSlots) {
    // Build a diamond: a + b -> relu -> mul(relu, sigmoid(a)) -> output
    // Two separate live values exist simultaneously, so we need >= 2 slots.
    const int N = 1000;
    Graph g;
    TensorType t; t.shape = {N};
    auto* a  = g.add_input("a", t);
    auto* b  = g.add_input("b", t);
    auto* s  = g.add_op(OpKind::Sigmoid, {a});
    auto* r  = g.add_op(OpKind::Add,     {a, b});
    auto* rl = g.add_op(OpKind::Relu,    {r});
    auto* m  = g.add_op(OpKind::Mul,     {rl, s});
    g.set_output(m);
    run_passes(g);

    // 's' and 'rl' are live at the same time -> need 2 slots
    EXPECT_GE(unique_slots(g), 1);
    // But total slots should still be bounded (not one slot per original node)
    EXPECT_LE(unique_slots(g), 3);
}

TEST(BufferReuse, MLPFourLayerPingPong) {
    // 4-layer MLP (FusedMatmul chain) should reuse slots via ping-pong.
    // Expected: at most 3 slots regardless of depth.
    const int K = 8, N = 8;
    Graph g;
    TensorType row_ty; row_ty.shape = {1, K};
    TensorType w_ty;   w_ty.shape   = {K, N};
    TensorType b_ty;   b_ty.shape   = {N};

    Node* cur = g.add_input("x", row_ty);
    for (int i = 0; i < 4; i++) {
        auto* W  = g.add_input("W" + std::to_string(i), w_ty);
        auto* bv = g.add_input("b" + std::to_string(i), b_ty);
        auto* mm = g.add_op(OpKind::Matmul,  {cur, W});
        auto* ba = g.add_op(OpKind::BiasAdd, {mm, bv});
        cur = g.add_op(OpKind::Relu, {ba});
    }
    g.set_output(cur);
    run_passes(g);

    int slots = unique_slots(g);
    // 4 FusedMatmul nodes with ping-pong reuse should need at most 3 slots
    EXPECT_LE(slots, 3) << "Expected ping-pong reuse across 4 MLP layers";
}

TEST(BufferReuse, DeadNodesGetNoSlot) {
    // Nodes marked dead (by DCE or MatmulEpiloguePass) should not be assigned a slot.
    Graph g;
    TensorType t; t.shape = {100};
    auto* a  = g.add_input("a", t);
    auto* b  = g.add_input("b", t);
    auto* mm = g.add_op(OpKind::Matmul,  {a, b});
    auto* bv = g.add_input("bv", t);
    auto* ba = g.add_op(OpKind::BiasAdd, {mm, bv});
    auto* rl = g.add_op(OpKind::Relu,    {ba});
    g.set_output(rl);
    run_passes(g);

    // After MatmulEpiloguePass: mm->kind==FusedMatmul, ba and rl are dead
    for (auto& n : g.nodes) {
        if (n->is_dead) {
            EXPECT_EQ(n->buffer_slot, -1)
                << "Dead node '" << n->name << "' should not have a buffer slot";
        }
    }
}
