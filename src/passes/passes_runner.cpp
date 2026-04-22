#include "ts/passes.h"
#include <iostream>

namespace ts {

void run_passes(Graph& g, bool verbose) {
    // MatmulEpilogue must run before Fusion so it can capture Relu/Sigmoid/Tanh
    // activations directly (before FusionPass wraps them into FusedKernels).
    ConstantFoldPass{}.run(g);
    MatmulEpiloguePass{}.run(g);
    FusionPass{}.run(g);
    DCEPass{}.run(g);
    BufferReusePass{}.run(g);
}

} // namespace ts
