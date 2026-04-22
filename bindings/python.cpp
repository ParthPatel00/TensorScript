#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// CodegenResult holds unique_ptr<llvm::Module> — the Module type must be
// complete at the point where CodegenResult is destroyed in this TU.
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "ts/ir.h"
#include "ts/passes.h"
#include "ts/codegen.h"
#include "ts/jit.h"
#include "ts/runtime.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// A Python-callable wrapper around CompiledFunction.
struct PyCompiledFunction {
    std::shared_ptr<ts::CompiledFunction> fn;
    std::shared_ptr<ts::TensorScriptJIT> jit;  // keep alive

    py::array_t<float> call(py::args args) {
        std::vector<float*> ptrs;
        std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> bufs;
        for (auto& a : args) {
            bufs.push_back(py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(a));
            ptrs.push_back(const_cast<float*>(bufs.back().data()));
        }

        float* out_data = fn->execute(ptrs);
        int64_t n = fn->numel;

        // Return a numpy array that COPIES the result (safe — pool slot may be reused)
        auto result = py::array_t<float>((py::ssize_t)n);
        std::copy(out_data, out_data + n, result.mutable_data());
        return result;
    }
};

// Python Graph wrapper — holds the IR Graph and drives compilation.
struct PyGraph {
    ts::Graph g;

    PyGraph() = default;

    py::object input(const std::string& name, std::vector<int64_t> shape) {
        ts::TensorType t;
        t.shape = shape;
        ts::Node* n = g.add_input(name, t);
        return py::cast(reinterpret_cast<uintptr_t>(n));
    }

    py::object add_op(ts::OpKind kind, py::object a, py::object b = py::none()) {
        auto* na = reinterpret_cast<ts::Node*>(a.cast<uintptr_t>());
        std::vector<ts::Node*> ins = {na};
        if (!b.is_none()) ins.push_back(reinterpret_cast<ts::Node*>(b.cast<uintptr_t>()));
        ts::Node* n = g.add_op(kind, ins);
        return py::cast(reinterpret_cast<uintptr_t>(n));
    }

    void set_output(py::object n) {
        g.set_output(reinterpret_cast<ts::Node*>(n.cast<uintptr_t>()));
    }

    PyCompiledFunction compile(bool dump_ir = false) {
        // Run optimization passes
        ts::run_passes(g);

        // Dump before/after dot graphs
        g.to_dot("results/ir_after_fusion.dot");

        // LLVM codegen
        ts::LLVMCodegen codegen;
        auto cg = codegen.emit(g, dump_ir);

        // JIT
        auto jit = std::make_shared<ts::TensorScriptJIT>();
        jit->add_module(std::move(cg.ctx), std::move(cg.module));

        // Build runtime
        auto cf = std::make_shared<ts::CompiledFunction>(
            ts::build_runtime(g, cg.kernels, *jit));

        PyCompiledFunction pcf;
        pcf.fn  = cf;
        pcf.jit = jit;
        return pcf;
    }

    void dump() { g.dump(); }
    void to_dot(const std::string& path) { g.to_dot(path); }
};

PYBIND11_MODULE(tensorscript, m) {
    m.doc() = "TensorScript: a C++/LLVM ML compiler";

    py::class_<PyCompiledFunction>(m, "CompiledFunction")
        .def("__call__", &PyCompiledFunction::call);

    py::class_<PyGraph>(m, "Graph")
        .def(py::init<>())
        .def("input", &PyGraph::input, py::arg("name"), py::arg("shape"))
        .def("set_output", &PyGraph::set_output)
        .def("compile", &PyGraph::compile, py::arg("dump_ir") = false)
        .def("dump", &PyGraph::dump)
        .def("to_dot", &PyGraph::to_dot)
        // Ops
        .def("add",     [](PyGraph& self, py::object a, py::object b) {
            return self.add_op(ts::OpKind::Add, a, b); })
        .def("sub",     [](PyGraph& self, py::object a, py::object b) {
            return self.add_op(ts::OpKind::Sub, a, b); })
        .def("mul",     [](PyGraph& self, py::object a, py::object b) {
            return self.add_op(ts::OpKind::Mul, a, b); })
        .def("div",     [](PyGraph& self, py::object a, py::object b) {
            return self.add_op(ts::OpKind::Div, a, b); })
        .def("relu",    [](PyGraph& self, py::object a) {
            return self.add_op(ts::OpKind::Relu, a); })
        .def("sigmoid", [](PyGraph& self, py::object a) {
            return self.add_op(ts::OpKind::Sigmoid, a); })
        .def("tanh",    [](PyGraph& self, py::object a) {
            return self.add_op(ts::OpKind::Tanh, a); })
        .def("exp",     [](PyGraph& self, py::object a) {
            return self.add_op(ts::OpKind::Exp, a); })
        .def("log",     [](PyGraph& self, py::object a) {
            return self.add_op(ts::OpKind::Log, a); })
        .def("neg",     [](PyGraph& self, py::object a) {
            return self.add_op(ts::OpKind::Neg, a); })
        .def("sqrt",    [](PyGraph& self, py::object a) {
            return self.add_op(ts::OpKind::Sqrt, a); })
        .def("matmul",  [](PyGraph& self, py::object a, py::object b) {
            return self.add_op(ts::OpKind::Matmul, a, b); })
        .def("bias_add",[](PyGraph& self, py::object a, py::object bias) {
            return self.add_op(ts::OpKind::BiasAdd, a, bias); });
}
