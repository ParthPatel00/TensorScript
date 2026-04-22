#include "ts/passes.h"
#include <iostream>

namespace ts {

void run_passes(Graph& g, bool verbose) {
    // 1. Constant folding
    ConstantFoldPass{}.run(g);
    // 2. Absorb Matmul+BiasAdd+Activation into FusedMatmul (must precede FusionPass)
    MatmulEpiloguePass{}.run(g);
    // 3. Replace remaining Sigmoid/Tanh/Exp/Log with VecLibCall on Apple Silicon
    //    (no-op on Linux; FusionPass then fuses across the remaining cheap ops)
    VecLibSplitPass{}.run(g);
    // 4. Fuse adjacent element-wise ops into FusedKernel (VecLibCall acts as a barrier)
    FusionPass{}.run(g);
    // 5. Dead code elimination
    DCEPass{}.run(g);
    // 6. Liveness-based buffer slot assignment
    BufferReusePass{}.run(g);
}

} // namespace ts
