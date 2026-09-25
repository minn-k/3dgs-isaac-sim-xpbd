cmake_minimum_required(VERSION 3.22)
project(gaussian_xpbd_isaac LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")
endif()

add_subdirectory(solver_core)

add_library(xpbd_isaac SHARED xpbd_capi.cu)
target_include_directories(xpbd_isaac PRIVATE solver_core/cuda_rasterizer solver_core/third_party/glm)
target_link_libraries(xpbd_isaac PRIVATE apg_xpbd_core)
set_target_properties(xpbd_isaac PROPERTIES
  OUTPUT_NAME xpbd_isaac
  CUDA_RUNTIME_LIBRARY Static
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
)
if(MSVC)
  target_compile_options(xpbd_isaac PRIVATE /utf-8)
endif()
