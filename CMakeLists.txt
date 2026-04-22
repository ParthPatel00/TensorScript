cmake_minimum_required(VERSION 3.20)
project(TensorScript VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── LLVM ──────────────────────────────────────────────────────────────────────
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(LLVM_LIBS
    core support orcjit native nativecodegen passes analysis transformutils
    vectorize instcombine scalaropts ipo mc
)

# ── BLAS ──────────────────────────────────────────────────────────────────────
if(APPLE)
    find_library(BLAS_LIB Accelerate REQUIRED)
    set(BLAS_INCLUDE_DIRS "")
else()
    find_package(BLAS REQUIRED)
    # Try OpenBLAS header location
    find_path(CBLAS_INCLUDE cblas.h
        PATHS /usr/include/openblas /usr/local/include/openblas
              /usr/include /usr/local/include
    )
    set(BLAS_INCLUDE_DIRS ${CBLAS_INCLUDE})
endif()

# ── Core library ──────────────────────────────────────────────────────────────
add_library(ts_core STATIC
    src/ir.cpp
    src/ops.cpp
    src/codegen.cpp
    src/jit.cpp
    src/runtime.cpp
    src/passes/constant_fold.cpp
    src/passes/fusion.cpp
    src/passes/matmul_epilogue.cpp
    src/passes/dce.cpp
    src/passes/buffer_reuse.cpp
    src/passes/passes_runner.cpp
)

target_include_directories(ts_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${BLAS_INCLUDE_DIRS}
)

target_link_libraries(ts_core PUBLIC ${LLVM_LIBS})

if(APPLE)
    target_link_libraries(ts_core PUBLIC ${BLAS_LIB})
else()
    target_link_libraries(ts_core PUBLIC ${BLAS_LIBRARIES})
endif()

# Aggressive optimization flags for the library itself
target_compile_options(ts_core PRIVATE -O3 -march=native -fno-rtti)

# ── Python bindings ───────────────────────────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(tensorscript bindings/python.cpp)
target_link_libraries(tensorscript PRIVATE ts_core)
# pybind11 3.x requires RTTI (typeid); do NOT pass -fno-rtti here.
target_compile_options(tensorscript PRIVATE -O3)

# ── C++ unit tests ────────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()

foreach(test_name test_ir test_fusion test_codegen test_matmul_epilogue test_buffer_reuse)
    if(EXISTS ${CMAKE_SOURCE_DIR}/tests/${test_name}.cpp)
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} ts_core GTest::gtest_main)
        target_compile_options(${test_name} PRIVATE -O2 -fno-rtti)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endforeach()
