cmake_minimum_required(VERSION 3.18)
project(libstable_diffusion_core.so C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- ExecuTorch SDK ---
# Set this to your ExecuTorch install/build directory
if(NOT DEFINED EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT "" CACHE PATH "Path to ExecuTorch source/build root")
endif()

if(NOT EXECUTORCH_ROOT)
  message(FATAL_ERROR "EXECUTORCH_ROOT must be set. Point it to your ExecuTorch build directory.")
endif()

set(OUTPUT_SUFFIX ${CMAKE_ANDROID_ARCH_ABI})
set(PLATFORM_LIBS log android dl m c)
# ANDROID_TOOLCHAIN_ROOT is set by the NDK toolchain file automatically

add_compile_options(
    -O3
    -Wall
    -fno-rtti
    -fPIC
    -fvisibility=hidden
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=for-loop-analysis")

# Skip missing cmake_macros.h (generated header not needed for ExecuTorch on Android)
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

# --- ExecuTorch includes and libraries ---
# These paths assume a standard ExecuTorch CMake build output structure.
# Adjust as needed based on your ExecuTorch build configuration.

# EXECUTORCH_ROOT is cmake-out-android; source root is two levels up
set(EXECUTORCH_SRC_ROOT ${EXECUTORCH_ROOT}/..)
set(EXECUTORCH_INCLUDE_DIRS
    ${EXECUTORCH_SRC_ROOT}/..
    ${EXECUTORCH_SRC_ROOT}
    ${EXECUTORCH_SRC_ROOT}/runtime/core/portable_type/c10
    ${EXECUTORCH_SRC_ROOT}/third-party/flatbuffers/include
    ${EXECUTORCH_ROOT}/include
    ${EXECUTORCH_ROOT}/sdk/include
)

# ExecuTorch static libraries — use direct paths (find_library fails in cross-compilation)
# Order matters: dependents before dependencies
set(ET_LIBS
    ${EXECUTORCH_ROOT}/extension/module/libextension_module_static.a
    ${EXECUTORCH_ROOT}/extension/tensor/libextension_tensor.a
    ${EXECUTORCH_ROOT}/extension/data_loader/libextension_data_loader.a
    ${EXECUTORCH_ROOT}/extension/named_data_map/libextension_named_data_map.a
    ${EXECUTORCH_ROOT}/extension/flat_tensor/libextension_flat_tensor.a
    ${EXECUTORCH_ROOT}/extension/threadpool/libextension_threadpool.a
    ${EXECUTORCH_ROOT}/backends/xnnpack/libxnnpack_backend.a
    ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/XNNPACK/libXNNPACK.a
    ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/XNNPACK/libxnnpack-microkernels-prod.a
    ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/libcpuinfo.a
    ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/libcpuinfo_internals.a
    ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/libpthreadpool.a
    ${EXECUTORCH_ROOT}/kernels/portable/libportable_ops_lib.a
    ${EXECUTORCH_ROOT}/kernels/portable/libportable_kernels.a
    ${EXECUTORCH_ROOT}/libexecutorch.a
    ${EXECUTORCH_ROOT}/libexecutorch_core.a
    ${EXECUTORCH_ROOT}/kleidiai/libkleidiai.a
)

# Verify all required libs exist
foreach(_lib ${ET_LIBS})
    if(NOT EXISTS ${_lib})
        message(WARNING "ExecuTorch lib not found: ${_lib}")
    endif()
endforeach()

# --- Third-party libraries (kept from original) ---

set(PACKAGE_INCLUDES
    ${EXECUTORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/json/include/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/json/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/stb
)

# tokenizers-cpp (for Qwen3 tokenizer)
set(TOKENZIER_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tokenizers-cpp)

if(NOT DEFINED CMAKE_ANDROID_ARCH_ABI)
    set(CMAKE_ANDROID_ARCH_ABI "arm64-v8a" CACHE STRING "Android ABI" FORCE)
endif()
if(NOT DEFINED ANDROID_ABI)
    set(ANDROID_ABI ${CMAKE_ANDROID_ARCH_ABI} CACHE STRING "Android ABI" FORCE)
endif()

add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

# --- Source files ---
file(GLOB SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main_flux2.cpp"
)

add_definitions(-DQNN_API=__attribute__\(\(visibility\(\"default\"\)\)\))

# --- Executable ---
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_options(${PROJECT_NAME} PRIVATE "-Wl,-s")
target_include_directories(${PROJECT_NAME} PRIVATE ${PACKAGE_INCLUDES})

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${OUTPUT_SUFFIX}"
)

# --- Link ---
# Kernel and backend libs need --whole-archive for static registration
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${PLATFORM_LIBS}
    ${EXECUTORCH_ROOT}/extension/module/libextension_module_static.a
    ${EXECUTORCH_ROOT}/extension/tensor/libextension_tensor.a
    ${EXECUTORCH_ROOT}/extension/data_loader/libextension_data_loader.a
    ${EXECUTORCH_ROOT}/extension/named_data_map/libextension_named_data_map.a
    ${EXECUTORCH_ROOT}/extension/flat_tensor/libextension_flat_tensor.a
    ${EXECUTORCH_ROOT}/extension/threadpool/libextension_threadpool.a
    -Wl,--whole-archive
    ${EXECUTORCH_ROOT}/backends/xnnpack/libxnnpack_backend.a
    ${EXECUTORCH_ROOT}/kleidiai/libkleidiai.a
    ${EXECUTORCH_ROOT}/configurations/liboptimized_native_cpu_ops_lib.a
    ${EXECUTORCH_ROOT}/kernels/quantized/libquantized_ops_lib.a
    ${EXECUTORCH_ROOT}/kernels/quantized/libquantized_kernels.a
    -Wl,--no-whole-archive
    ${EXECUTORCH_ROOT}/kernels/optimized/liboptimized_kernels.a
    ${EXECUTORCH_ROOT}/kernels/optimized/libcpublas.a
    ${EXECUTORCH_ROOT}/kernels/optimized/libeigen_blas.a
    ${EXECUTORCH_ROOT}/kernels/portable/liboptimized_portable_kernels.a
    ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/XNNPACK/libXNNPACK.a
    ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/XNNPACK/libxnnpack-microkernels-prod.a
    ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/libcpuinfo.a
    ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/libcpuinfo_internals.a
    ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/libpthreadpool.a
    ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/libkernels_util_all_deps.a
    ${EXECUTORCH_ROOT}/libexecutorch.a
    ${EXECUTORCH_ROOT}/libexecutorch_core.a
    tokenizers_cpp
)
