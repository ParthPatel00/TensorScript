#pragma once
#include "ts/ir.h"

namespace ts {

struct PassBase {
    virtual ~PassBase() = default;
    virtual void run(Graph& g) = 0;
    virtual const char* name() const = 0;
};

struct ConstantFoldPass : PassBase {
    void run(Graph& g) override;
    const char* name() const override { return "ConstantFold"; }
};

struct FusionPass : PassBase {
    void run(Graph& g) override;
    const char* name() const override { return "FusionPass"; }
};

struct MatmulEpiloguePass : PassBase {
    void run(Graph& g) override;
    const char* name() const override { return "MatmulEpiloguePass"; }
};

struct DCEPass : PassBase {
    void run(Graph& g) override;
    const char* name() const override { return "DCEPass"; }
};

struct BufferReusePass : PassBase {
    void run(Graph& g) override;
    const char* name() const override { return "BufferReusePass"; }
    int num_slots = 0;
};

void run_passes(Graph& g, bool verbose = false);

} // namespace ts
