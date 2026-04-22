#pragma once
#include <cstdint>
#include <memory>
#include <string>

namespace llvm { class Module; class LLVMContext; }
namespace llvm::orc { class LLJIT; }

namespace ts {

// Kernel function signature: inputs array, num_inputs, output, element count
using KernelFn = void(*)(float**, int, float*, int64_t);

class FuseJIT {
public:
    FuseJIT();
    ~FuseJIT();

    // Takes ownership of module + ctx; compiles with O3
    void add_module(std::unique_ptr<llvm::LLVMContext> ctx,
                    std::unique_ptr<llvm::Module> module);

    KernelFn lookup(const std::string& name);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ts
