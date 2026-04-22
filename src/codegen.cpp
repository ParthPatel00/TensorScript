#include "ts/codegen.h"
#include "ts/ir.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Config/llvm-config.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

namespace ts {

using namespace llvm;

// getOrInsertDeclaration was introduced in LLVM 20; use getDeclaration on older versions.
static Function* get_intrinsic(Module* m, Intrinsic::ID id, Type* ty) {
#if LLVM_VERSION_MAJOR >= 20
    return Intrinsic::getOrInsertDeclaration(m, id, {ty});
#else
    return Intrinsic::getDeclaration(m, id, {ty});
#endif
}

// Emit one element of sigmoid: 1 / (1 + exp(-x))
static Value* emit_sigmoid(IRBuilder<>& b, Value* x, Module* m) {
    auto* ft = Type::getFloatTy(b.getContext());
    auto* neg_x = b.CreateFNeg(x, "neg_x");
    auto* ex = b.CreateCall(get_intrinsic(m, Intrinsic::exp, ft), {neg_x}, "exp_neg");
    auto* one = ConstantFP::get(ft, 1.0);
    auto* denom = b.CreateFAdd(one, ex, "denom");
    return b.CreateFDiv(one, denom, "sigmoid");
}

// Emit one element of tanh: (exp(2x)-1)/(exp(2x)+1)
static Value* emit_tanh(IRBuilder<>& b, Value* x, Module* m) {
    auto* ft = Type::getFloatTy(b.getContext());
    auto* two = ConstantFP::get(ft, 2.0);
    auto* two_x = b.CreateFMul(two, x, "two_x");
    auto* e = b.CreateCall(get_intrinsic(m, Intrinsic::exp, ft), {two_x}, "exp2x");
    auto* one = ConstantFP::get(ft, 1.0);
    auto* num = b.CreateFSub(e, one, "tanh_num");
    auto* den = b.CreateFAdd(e, one, "tanh_den");
    return b.CreateFDiv(num, den, "tanh");
}

// Apply a single FusedOp to the accumulator. inputs[i] are the loaded element values.
static Value* emit_fused_op(IRBuilder<>& b, const FusedOp& op,
                             Value* acc, const std::vector<Value*>& input_vals,
                             Module* m) {
    auto* ft = Type::getFloatTy(b.getContext());
    Value* other = (op.input_idx >= 0) ? input_vals[op.input_idx] : nullptr;

    switch (op.kind) {
    case OpKind::Add:  return b.CreateFAdd(acc, other, "add");
    case OpKind::Sub:  return b.CreateFSub(acc, other, "sub");
    case OpKind::Mul:  return b.CreateFMul(acc, other, "mul");
    case OpKind::Div:  return b.CreateFDiv(acc, other, "div");
    case OpKind::Neg:  return b.CreateFNeg(acc, "neg");
    case OpKind::Relu: {
        auto* zero = ConstantFP::get(ft, 0.0);
        return b.CreateCall(get_intrinsic(m, Intrinsic::maxnum, ft), {acc, zero}, "relu");
    }
    case OpKind::Exp:
        return b.CreateCall(get_intrinsic(m, Intrinsic::exp,  ft), {acc}, "exp");
    case OpKind::Log:
        return b.CreateCall(get_intrinsic(m, Intrinsic::log,  ft), {acc}, "log");
    case OpKind::Sqrt:
        return b.CreateCall(get_intrinsic(m, Intrinsic::sqrt, ft), {acc}, "sqrt");
    case OpKind::Sigmoid: return emit_sigmoid(b, acc, m);
    case OpKind::Tanh:    return emit_tanh(b, acc, m);
    default:
        assert(false && "unknown op in emit_fused_op");
        return acc;
    }
}

// Emit a FusedKernel function:
//   void @name(float** inputs, i32 num_inputs, float* output, i64 n)
static Function* emit_fused_kernel(Module* m, const std::string& name, const Node* node) {
    LLVMContext& ctx = m->getContext();
    auto* ft      = Type::getFloatTy(ctx);
    auto* i64_ty  = Type::getInt64Ty(ctx);
    auto* i32_ty  = Type::getInt32Ty(ctx);
    auto* ptr_ty  = PointerType::getUnqual(ft);
    auto* pptr_ty = PointerType::getUnqual(ptr_ty);
    auto* void_ty = Type::getVoidTy(ctx);

    auto* fn_ty = FunctionType::get(void_ty, {pptr_ty, i32_ty, ptr_ty, i64_ty}, false);
    auto* fn = Function::Create(fn_ty, Function::ExternalLinkage, name, m);
    fn->setDoesNotThrow();
    // noalias lets the auto-vectorizer assume no pointer aliasing across args.
    // BufferReusePass enforces this invariant at compile time.
    fn->addParamAttr(0, Attribute::NoAlias);  // inputs_ptr (float**)
    fn->addParamAttr(2, Attribute::NoAlias);  // out_ptr (float*)

    auto args = fn->arg_begin();
    auto* inputs_ptr = &*args++;  inputs_ptr->setName("inputs");
    auto* /*num_in*/ _ = &*args++;
    auto* out_ptr   = &*args++;  out_ptr->setName("out");
    auto* n_arg     = &*args;    n_arg->setName("n");

    (void)_;  // unused but declared for ABI

    auto* entry_bb = BasicBlock::Create(ctx, "entry", fn);
    auto* preheader_bb = BasicBlock::Create(ctx, "preheader", fn);
    auto* loop_bb  = BasicBlock::Create(ctx, "loop", fn);
    auto* exit_bb  = BasicBlock::Create(ctx, "exit", fn);

    IRBuilder<> b(ctx);

    // entry: check n > 0
    b.SetInsertPoint(entry_bb);
    auto* cmp_entry = b.CreateICmpEQ(n_arg, ConstantInt::get(i64_ty, 0), "empty");
    b.CreateCondBr(cmp_entry, exit_bb, preheader_bb);

    // preheader: load base pointers for each input array
    b.SetInsertPoint(preheader_bb);
    std::vector<Value*> base_ptrs;
    for (int k = 0; k < (int)node->inputs.size(); k++) {
        auto* idx_v = ConstantInt::get(i32_ty, k);
        auto* slot_ptr = b.CreateGEP(ptr_ty, inputs_ptr, idx_v, "slot");
        auto* base = b.CreateLoad(ptr_ty, slot_ptr, "base" + std::to_string(k));
        base_ptrs.push_back(base);
    }
    b.CreateBr(loop_bb);

    // loop
    b.SetInsertPoint(loop_bb);
    auto* i_phi = b.CreatePHI(i64_ty, 2, "i");
    i_phi->addIncoming(ConstantInt::get(i64_ty, 0), preheader_bb);

    // Load one float from each input
    std::vector<Value*> input_vals;
    for (int k = 0; k < (int)base_ptrs.size(); k++) {
        auto* gep = b.CreateGEP(ft, base_ptrs[k], i_phi, "gep_in" + std::to_string(k));
        auto* v = b.CreateLoad(ft, gep, "in" + std::to_string(k));
        input_vals.push_back(v);
    }

    // Accumulator starts as input[0]
    Value* acc = input_vals.empty() ? ConstantFP::get(ft, 0.0) : input_vals[0];

    // Apply each fused op
    for (auto& fop : node->fused_ops) {
        acc = emit_fused_op(b, fop, acc, input_vals, m);
    }

    // Store result
    auto* out_gep = b.CreateGEP(ft, out_ptr, i_phi, "gep_out");
    b.CreateStore(acc, out_gep);

    // Loop increment and branch
    auto* i_next = b.CreateAdd(i_phi, ConstantInt::get(i64_ty, 1), "i_next");
    i_phi->addIncoming(i_next, loop_bb);
    auto* done = b.CreateICmpEQ(i_next, n_arg, "done");
    b.CreateCondBr(done, exit_bb, loop_bb);

    b.SetInsertPoint(exit_bb);
    b.CreateRetVoid();

    // Add loop vectorize metadata
    MDNode* loop_md = MDNode::get(ctx, {
        MDString::get(ctx, "llvm.loop.vectorize.enable"),
        ConstantAsMetadata::get(ConstantInt::getTrue(ctx))
    });
    auto* loop_id = MDNode::get(ctx, {loop_md});
    loop_bb->getTerminator()->setMetadata("llvm.loop", loop_id);

    return fn;
}

// Emit a FusedMatmul kernel.
// For batch=1 (output shape [M, N] from [1, K] x [K, N]):
//   We call into a BLAS GEMV shim declared as an external function,
//   then fuse the epilogue (bias add + activation) in a scalar loop.
// For batch>1: same BLAS shim (sgemm) + epilogue loop.
//
// Epilogue function signature:
//   void @name(float* matmul_out, float* bias_or_null, float* final_out, i64 n)
// (The actual GEMM call is done in the C++ runtime before calling the epilogue kernel.)
static Function* emit_matmul_epilogue(Module* m, const std::string& name,
                                       const Node* node) {
    LLVMContext& ctx = m->getContext();
    auto* ft     = Type::getFloatTy(ctx);
    auto* i64_ty = Type::getInt64Ty(ctx);
    auto* ptr_ty = PointerType::getUnqual(ft);
    auto* void_ty = Type::getVoidTy(ctx);
    auto* i1_ty  = Type::getInt1Ty(ctx);

    // Args: matmul_out, bias (null if no bias), final_out, n, has_bias
    auto* fn_ty = FunctionType::get(void_ty, {ptr_ty, ptr_ty, ptr_ty, i64_ty, i1_ty}, false);
    auto* fn = Function::Create(fn_ty, Function::ExternalLinkage, name, m);
    fn->setDoesNotThrow();

    auto args = fn->arg_begin();
    auto* mm_out   = &*args++;  mm_out->setName("mm_out");
    auto* bias_ptr = &*args++;  bias_ptr->setName("bias");
    auto* final_out = &*args++; final_out->setName("final_out");
    auto* n_arg     = &*args++; n_arg->setName("n");
    auto* has_bias  = &*args;   has_bias->setName("has_bias");

    auto* entry_bb = BasicBlock::Create(ctx, "entry", fn);
    auto* loop_bb  = BasicBlock::Create(ctx, "loop", fn);
    auto* exit_bb  = BasicBlock::Create(ctx, "exit", fn);

    IRBuilder<> b(ctx);
    b.SetInsertPoint(entry_bb);
    auto* cmp_entry = b.CreateICmpEQ(n_arg, ConstantInt::get(i64_ty, 0));
    b.CreateCondBr(cmp_entry, exit_bb, loop_bb);

    b.SetInsertPoint(loop_bb);
    auto* i_phi = b.CreatePHI(i64_ty, 2, "i");
    i_phi->addIncoming(ConstantInt::get(i64_ty, 0), entry_bb);

    auto* mm_gep = b.CreateGEP(ft, mm_out, i_phi);
    auto* v = b.CreateLoad(ft, mm_gep, "v");

    // Bias add (conditional on has_bias at element level — always true/false per call)
    auto* bias_gep = b.CreateGEP(ft, bias_ptr, i_phi);
    auto* bias_v = b.CreateLoad(ft, bias_gep, "bias_v");
    auto* v_biased = b.CreateSelect(has_bias, b.CreateFAdd(v, bias_v, "biased"), v, "v_b");

    // Activation
    Value* result = v_biased;
    OpKind ep = node->epilogue;
    if (ep == OpKind::Relu) {
        auto* zero = ConstantFP::get(ft, 0.0);
        result = b.CreateCall(get_intrinsic(m, Intrinsic::maxnum, ft), {v_biased, zero}, "relu");
    } else if (ep == OpKind::Sigmoid) {
        result = emit_sigmoid(b, v_biased, m);
    } else if (ep == OpKind::Tanh) {
        result = emit_tanh(b, v_biased, m);
    }

    auto* out_gep = b.CreateGEP(ft, final_out, i_phi);
    b.CreateStore(result, out_gep);

    auto* i_next = b.CreateAdd(i_phi, ConstantInt::get(i64_ty, 1), "i_next");
    i_phi->addIncoming(i_next, loop_bb);
    b.CreateCondBr(b.CreateICmpEQ(i_next, n_arg, "done"), exit_bb, loop_bb);

    b.SetInsertPoint(exit_bb);
    b.CreateRetVoid();

    return fn;
}

CodegenResult LLVMCodegen::emit(const Graph& g, bool dump_ir) {
    auto ctx    = std::make_unique<LLVMContext>();
    auto module = std::make_unique<Module>("fuse", *ctx);

    std::vector<std::pair<std::string, Node*>> kernels;
    int kid = 0;

    auto order = g.topo_order();
    for (auto* node : order) {
        if (node->is_dead) continue;
        if (node->kind == OpKind::FusedKernel) {
            std::string kname = "kernel_" + std::to_string(kid++);
            emit_fused_kernel(module.get(), kname, node);
            kernels.push_back({kname, node});
        } else if (node->kind == OpKind::FusedMatmul) {
            std::string kname = "epilogue_" + std::to_string(kid++);
            emit_matmul_epilogue(module.get(), kname, node);
            kernels.push_back({kname, node});
        }
        // VecLibCall: no LLVM IR generated — dispatched directly by runtime.cpp
    }

    // Verify
    std::string err;
    raw_string_ostream os(err);
    if (verifyModule(*module, &os)) {
        std::cerr << "[Fuse] LLVM module verification failed:\n" << err << "\n";
    }

    if (dump_ir) {
        std::string ir_str;
        raw_string_ostream iros(ir_str);
        module->print(iros, nullptr);
        std::ofstream f("results/kernel_scalar.ll");
        f << ir_str;
        std::cout << "[Fuse] Wrote results/kernel_scalar.ll\n";
    }

    return {std::move(ctx), std::move(module), std::move(kernels)};
}

} // namespace ts
