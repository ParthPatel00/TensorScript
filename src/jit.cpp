#include "ts/jit.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace ts {

struct FuseJIT::Impl {
    std::unique_ptr<llvm::orc::LLJIT> jit;
};

FuseJIT::FuseJIT() : impl_(std::make_unique<Impl>()) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    auto jit = llvm::orc::LLJITBuilder().create();
    if (!jit) {
        std::string err;
        llvm::raw_string_ostream os(err);
        os << jit.takeError();
        throw std::runtime_error("Failed to create LLJIT: " + err);
    }
    impl_->jit = std::move(*jit);
}

FuseJIT::~FuseJIT() = default;

static void optimize_module(llvm::Module& m) {
    llvm::PassBuilder pb;
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm =
        pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    mpm.run(m, mam);
}

void FuseJIT::add_module(std::unique_ptr<llvm::LLVMContext> ctx,
                                  std::unique_ptr<llvm::Module> module) {
    optimize_module(*module);

    // Dump vectorized IR
    {
        std::string ir;
        llvm::raw_string_ostream os(ir);
        module->print(os, nullptr);
        // Write to results/ if directory exists
        std::ofstream f("results/kernel_vectorized.ll");
        if (f) f << ir;
    }

    auto tsm = llvm::orc::ThreadSafeModule(std::move(module), std::move(ctx));
    auto err = impl_->jit->addIRModule(std::move(tsm));
    if (err) {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << std::move(err);
        throw std::runtime_error("JIT addIRModule failed: " + msg);
    }
    std::cout << "[Fuse] JIT: module compiled\n";
}

KernelFn FuseJIT::lookup(const std::string& name) {
    auto sym = impl_->jit->lookup(name);
    if (!sym) {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << sym.takeError();
        throw std::runtime_error("JIT lookup failed for '" + name + "': " + msg);
    }
    return reinterpret_cast<KernelFn>(sym->getValue());
}

} // namespace ts
