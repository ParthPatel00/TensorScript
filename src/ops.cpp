#include "ts/ops.h"
#include <stdexcept>

namespace ts {

static const OpInfo kOpTable[] = {
    {OpKind::Add,     true,  true,  ""},
    {OpKind::Sub,     true,  true,  ""},
    {OpKind::Mul,     true,  true,  ""},
    {OpKind::Div,     true,  true,  ""},
    {OpKind::Relu,    true,  false, "llvm.maxnum.f32"},
    {OpKind::Sigmoid, true,  false, ""},   // computed inline: 1/(1+exp(-x))
    {OpKind::Tanh,    true,  false, ""},   // computed inline: (exp(2x)-1)/(exp(2x)+1)
    {OpKind::Exp,     true,  false, "llvm.exp.f32"},
    {OpKind::Log,     true,  false, "llvm.log.f32"},
    {OpKind::Neg,     true,  false, ""},
    {OpKind::Sqrt,    true,  false, "llvm.sqrt.f32"},
};

const OpInfo& get_op_info(OpKind kind) {
    for (auto& info : kOpTable) {
        if (info.kind == kind) return info;
    }
    throw std::runtime_error("get_op_info: unknown op");
}

} // namespace ts
