"""
Numerical correctness gate: every op and common chain must match NumPy to atol=1e-5.
Run BEFORE any benchmarking.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import fuse as ts

ATOL = 1e-5
N = 100_000
rng = np.random.default_rng(42)

def make_inputs(n=N):
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    # Clip to avoid exp overflow
    a = np.clip(a, -5, 5)
    b = np.clip(b, -5, 5)
    return a, b

def compile_chain(ops, n=N):
    """Build and compile a graph applying ops in sequence to input(s)."""
    g = ts.Graph()
    a = g.input("a", [n])
    b = g.input("b", [n])
    x = a
    for op_name in ops:
        op_fn = getattr(g, op_name)
        if op_name in ("add", "sub", "mul", "div"):
            x = op_fn(x, b)
        else:
            x = op_fn(x)
    g.set_output(x)
    return g.compile(), b

def check(ts_out, np_ref, label):
    close = np.allclose(ts_out, np_ref, atol=ATOL)
    status = "PASS" if close else "FAIL"
    maxdiff = np.max(np.abs(ts_out - np_ref))
    print(f"  [{status}] {label:40s}  max_diff={maxdiff:.2e}")
    if not close:
        idx = np.argmax(np.abs(ts_out - np_ref))
        print(f"         first bad: ts={ts_out[idx]:.6f}  np={np_ref[idx]:.6f}  i={idx}")
    return close

def test_single_ops():
    print("\n=== Single ops ===")
    a, b = make_inputs()
    passed = 0
    tests = [
        ("add",     lambda a,b: a + b),
        ("sub",     lambda a,b: a - b),
        ("mul",     lambda a,b: a * b),
        ("div",     lambda a,b: a / b),
        ("relu",    lambda a,b: np.maximum(a, 0)),
        ("sigmoid", lambda a,b: 1/(1+np.exp(-a))),
        ("tanh",    lambda a,b: np.tanh(a)),
        ("exp",     lambda a,b: np.exp(a)),
        ("log",     lambda a,b: np.log(a)),  # test_a is already np.abs(a)+0.01 > 0
        ("neg",     lambda a,b: -a),
        ("sqrt",    lambda a,b: np.sqrt(np.abs(a))),
    ]
    for op_name, ref_fn in tests:
        if op_name in ("log", "sqrt"):
            test_a = np.abs(a) + 0.01
        else:
            test_a = a
        fn, _ = compile_chain([op_name])
        ts_out = fn(test_a, b)
        np_ref = ref_fn(test_a, b).astype(np.float32)
        if check(ts_out, np_ref, op_name): passed += 1
    return passed, len(tests)

def test_chains():
    print("\n=== Chains ===")
    a, b = make_inputs()
    passed = 0
    tests = [
        ("add→relu",         ["add", "relu"],           lambda a,b: np.maximum(a+b, 0)),
        ("add→relu→mul",     ["add", "relu", "mul"],    lambda a,b: np.maximum(a+b,0)*b),
        ("add→sigmoid→tanh", ["add", "sigmoid", "tanh"],lambda a,b: np.tanh(1/(1+np.exp(-(a+b))))),
        ("relu→exp",         ["relu", "exp"],            lambda a,b: np.exp(np.maximum(a,0))),
        ("5-chain",          ["add","relu","mul","sigmoid","tanh"],
                              lambda a,b: np.tanh(1/(1+np.exp(-( np.maximum(a+b,0)*b ))))),
    ]
    for label, ops, ref_fn in tests:
        fn, _ = compile_chain(ops)
        ts_out = fn(a, b)
        np_ref = ref_fn(a, b).astype(np.float32)
        if check(ts_out, np_ref, label): passed += 1
    return passed, len(tests)

if __name__ == "__main__":
    p1, t1 = test_single_ops()
    p2, t2 = test_chains()
    total = t1 + t2
    passed = p1 + p2
    print(f"\n{'='*55}")
    print(f"Correctness: {passed}/{total} passed")
    if passed < total:
        print("BENCHMARKS MUST NOT RUN until all correctness tests pass.")
        sys.exit(1)
    else:
        print("All correctness tests passed. Benchmarks may proceed.")
