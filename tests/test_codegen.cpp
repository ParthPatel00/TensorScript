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
#include <numeric>
#include <vector>

using namespace ts;

static bool allclose(const std::vector<float>& a, const std::vector<float>& b,
                     float atol = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > atol) return false;
    }
    return true;
}

static std::vector<float> randn(int n) {
    std::vector<float> v(n);
    for (auto& x : v) x = (float)(rand() % 2000 - 1000) / 1000.f;
    return v;
}

TEST(Codegen, AddRelu) {
    const int N = 10000;
    Graph g;
    TensorType t; t.shape = {N};
    auto* a = g.add_input("a", t);
    auto* b = g.add_input("b", t);
    auto* x = g.add_op(OpKind::Add,  {a, b});
    auto* y = g.add_op(OpKind::Relu, {x});
    g.set_output(y);

    run_passes(g);

    LLVMCodegen cg;
    auto res = cg.emit(g, false);

    FuseJIT jit;
    jit.add_module(std::move(res.ctx), std::move(res.module));

    CompiledFunction cf = build_runtime(g, res.kernels, jit);

    auto av = randn(N), bv = randn(N);
    float* out = cf.execute({av.data(), bv.data()});

    // Compute NumPy-equivalent reference
    std::vector<float> ref(N);
    for (int i = 0; i < N; i++) ref[i] = std::max(0.f, av[i] + bv[i]);

    std::vector<float> got(out, out + N);
    EXPECT_TRUE(allclose(got, ref));
}

TEST(Codegen, SigmoidTanhChain) {
    const int N = 10000;
    Graph g;
    TensorType t; t.shape = {N};
    auto* a = g.add_input("a", t);
    auto* x0 = g.add_op(OpKind::Sigmoid, {a});
    auto* x1 = g.add_op(OpKind::Tanh,    {x0});
    g.set_output(x1);

    run_passes(g);

    LLVMCodegen cg;
    auto res = cg.emit(g, false);
    FuseJIT jit;
    jit.add_module(std::move(res.ctx), std::move(res.module));
    CompiledFunction cf = build_runtime(g, res.kernels, jit);

    auto av = randn(N);
    float* out = cf.execute({av.data()});

    std::vector<float> ref(N);
    for (int i = 0; i < N; i++) {
        float s = 1.f / (1.f + std::exp(-av[i]));
        ref[i] = std::tanh(s);
    }
    std::vector<float> got(out, out + N);
    EXPECT_TRUE(allclose(got, ref, 1e-4f));
}
