#pragma once
#include "ts/ir.h"

namespace ts {

struct OpInfo {
    OpKind kind;
    bool is_elementwise;
    bool is_binary;
    // LLVM intrinsic name for unary ops ("" if handled inline)
    const char* llvm_intrinsic;
};

const OpInfo& get_op_info(OpKind kind);

} // namespace ts
