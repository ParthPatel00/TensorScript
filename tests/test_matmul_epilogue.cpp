#include <gtest/gtest.h>
// CodegenResult holds unique_ptr<llvm::Module/LLVMContext>; types must be complete here.
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "ts/ir.h"
#include "ts/passes.h"
#include "ts/codegen.h"
#include "ts/jit.h"
#include "ts/runtime.h"

#include <cmath>
#include <vector>

using namespace ts;

static bool allclose(const float* got, const std::vector<float>& ref, float atol = 1e-4f) {
    for (size_t i = 0; i < ref.size(); i++) {
        if (std::abs(got[i] - ref[i]) > atol) return false;
    }
    return true;
}

// Reference GEMM: C[m][n] = sum_k A[m][k] * B[k][n]
static std::vector<float> ref_gemm(const float* A, const float* B,
                                    int M, int K, int N) {
    std::vector<float> C(M * N, 0.f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                C[m * N + n] += A[m * K + k] * B[k * N + n];
    return C;
}

TEST(MatmulEpilogue, PassFusesPattern) {
    // Verify MatmulEpiloguePass collapses Matmul → BiasAdd → Relu into
    // a single FusedMatmul node with epilogue=Relu.
    // MatmulEpilogue runs before FusionPass, so it captures the raw Relu node.
    Graph g;
    TensorType x_ty; x_ty.shape = {1, 4};
    TensorType w_ty; w_ty.shape = {4, 8};
    TensorType b_ty; b_ty.shape = {8};
    auto* x  = g.add_input("x", x_ty);
    auto* W  = g.add_input("W", w_ty);
    auto* bv = g.add_input("b", b_ty);
    auto* mm = g.add_op(OpKind::Matmul,  {x, W});
    auto* ba = g.add_op(OpKind::BiasAdd, {mm, bv});
    auto* rl = g.add_op(OpKind::Relu,    {ba});
    g.set_output(rl);

    run_passes(g);

    // After run_passes: mm upgraded to FusedMatmul, ba and rl are dead.
    EXPECT_EQ(mm->kind,     OpKind::FusedMatmul);
    EXPECT_EQ(mm->epilogue, OpKind::Relu);
    EXPECT_EQ(g.output,     mm);
    EXPECT_TRUE(ba->is_dead);
    EXPECT_TRUE(rl->is_dead);
}

TEST(MatmulEpilogue, ShapeInference) {
    // Matmul [1,4] x [4,8] -> output should be [1,8]
    Graph g;
    TensorType x_ty; x_ty.shape = {1, 4};
    TensorType w_ty; w_ty.shape = {4, 8};
    auto* x = g.add_input("x", x_ty);
    auto* W = g.add_input("W", w_ty);
    auto* mm = g.add_op(OpKind::Matmul, {x, W});
    g.set_output(mm);

    ASSERT_EQ(mm->output_type.shape.size(), 2u);
    EXPECT_EQ(mm->output_type.shape[0], 1);
    EXPECT_EQ(mm->output_type.shape[1], 8);
}

TEST(MatmulEpilogue, GEMVWithBiasRelu) {
    // [1,4] x [4,8] + bias[8] -> relu -> [1,8]
    const int M = 1, K = 4, N = 8;
    Graph g;
    TensorType x_ty; x_ty.shape = {M, K};
    TensorType w_ty; w_ty.shape = {K, N};
    TensorType b_ty; b_ty.shape = {N};
    auto* x  = g.add_input("x",  x_ty);
    auto* W  = g.add_input("W",  w_ty);
    auto* bv = g.add_input("bv", b_ty);
    auto* mm = g.add_op(OpKind::Matmul,  {x, W});
    auto* ba = g.add_op(OpKind::BiasAdd, {mm, bv});
    auto* rl = g.add_op(OpKind::Relu,    {ba});
    g.set_output(rl);

    run_passes(g);
    // After reordered passes, mm is upgraded to FusedMatmul with relu epilogue.
    ASSERT_EQ(mm->kind, OpKind::FusedMatmul);
    ASSERT_EQ(g.output, mm);

    LLVMCodegen cg;
    auto res = cg.emit(g, false);
    FuseJIT jit;
    jit.add_module(std::move(res.ctx), std::move(res.module));
    CompiledFunction cf = build_runtime(g, res.kernels, jit);

    std::vector<float> xv(M * K), wv(K * N), bvv(N);
    for (int i = 0; i < M * K; i++) xv[i] = (float)(i + 1) * 0.1f;
    for (int i = 0; i < K * N; i++) wv[i] = (float)(i % 5) * 0.2f - 0.4f;
    for (int i = 0; i < N;     i++) bvv[i] = (float)(i) * 0.05f - 0.2f;

    float* out = cf.execute({xv.data(), wv.data(), bvv.data()});

    auto gemm = ref_gemm(xv.data(), wv.data(), M, K, N);
    std::vector<float> ref(N);
    for (int i = 0; i < N; i++)
        ref[i] = std::max(0.f, gemm[i] + bvv[i]);

    EXPECT_TRUE(allclose(out, ref)) << "GEMVWithBiasRelu: numerical mismatch";
}

TEST(MatmulEpilogue, GEMVWithBiasSigmoid) {
    // [1,6] x [6,6] + bias[6] -> sigmoid
    const int M = 1, K = 6, N = 6;
    Graph g;
    TensorType x_ty; x_ty.shape = {M, K};
    TensorType w_ty; w_ty.shape = {K, N};
    TensorType b_ty; b_ty.shape = {N};
    auto* x  = g.add_input("x",  x_ty);
    auto* W  = g.add_input("W",  w_ty);
    auto* bv = g.add_input("bv", b_ty);
    auto* mm = g.add_op(OpKind::Matmul,  {x, W});
    auto* ba = g.add_op(OpKind::BiasAdd, {mm, bv});
    auto* sg = g.add_op(OpKind::Sigmoid, {ba});
    g.set_output(sg);

    run_passes(g);
    ASSERT_EQ(mm->kind, OpKind::FusedMatmul);
    ASSERT_EQ(g.output, mm);

    LLVMCodegen cg;
    auto res = cg.emit(g, false);
    FuseJIT jit;
    jit.add_module(std::move(res.ctx), std::move(res.module));
    CompiledFunction cf = build_runtime(g, res.kernels, jit);

    std::vector<float> xv(M * K), wv(K * N), bvv(N);
    for (int i = 0; i < M * K; i++) xv[i] = (float)(i - K/2) * 0.3f;
    for (int i = 0; i < K * N; i++) wv[i] = (float)((i * 3) % 7) * 0.1f - 0.3f;
    for (int i = 0; i < N;     i++) bvv[i] = (float)(i) * 0.1f - 0.25f;

    float* out = cf.execute({xv.data(), wv.data(), bvv.data()});

    auto gemm = ref_gemm(xv.data(), wv.data(), M, K, N);
    std::vector<float> ref(N);
    for (int i = 0; i < N; i++) {
        float v = gemm[i] + bvv[i];
        ref[i] = 1.f / (1.f + std::exp(-v));
    }

    EXPECT_TRUE(allclose(out, ref, 1e-4f)) << "GEMVWithBiasSigmoid: numerical mismatch";
}

TEST(MatmulEpilogue, MatmulNoBias) {
    // Matmul with no bias — only the matmul, no epilogue fusion
    // (pass requires Matmul->BiasAdd to trigger; bare Matmul is not fused)
    const int M = 1, K = 4, N = 4;
    Graph g;
    TensorType x_ty; x_ty.shape = {M, K};
    TensorType w_ty; w_ty.shape = {K, N};
    auto* x  = g.add_input("x", x_ty);
    auto* W  = g.add_input("W", w_ty);
    auto* mm = g.add_op(OpKind::Matmul, {x, W});
    g.set_output(mm);

    // Pass should not fuse a bare Matmul (needs at least BiasAdd)
    MatmulEpiloguePass mep;
    mep.run(g);
    // Still a plain Matmul
    EXPECT_EQ(g.output->kind, OpKind::Matmul);
}
