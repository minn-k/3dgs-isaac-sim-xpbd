cmake_minimum_required(VERSION 3.22)

add_library(apg_xpbd_core STATIC
  cuda_rasterizer/backward.h
  cuda_rasterizer/backward.cu
  cuda_rasterizer/forward.h
  cuda_rasterizer/forward.cu
  cuda_rasterizer/auxiliary.h
  cuda_rasterizer/rasterizer_impl.cu
  cuda_rasterizer/rasterizer_impl.h
  cuda_rasterizer/rasterizer.h
)

target_include_directories(apg_xpbd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(apg_xpbd_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(apg_xpbd_core PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
