#pragma once
#include "ts/ir.h"
#include <memory>
#include <string>

// Forward-declare LLVM types to avoid pulling all of LLVM into every TU
namespace llvm {
class Module;
class LLVMContext;
} // namespace llvm

namespace ts {

struct CodegenResult {
    std::unique_ptr<llvm::LLVMContext> ctx;
    std::unique_ptr<llvm::Module> module;
    // Ordered list of (kernel_name, node*) pairs matching topo order
    std::vector<std::pair<std::string, Node*>> kernels;
};

class LLVMCodegen {
public:
    // Emit LLVM IR for all kernel nodes in the graph.
    // Returns a Module ready to hand to OrcJIT.
    CodegenResult emit(const Graph& g, bool dump_ir = false);
};

} // namespace ts
