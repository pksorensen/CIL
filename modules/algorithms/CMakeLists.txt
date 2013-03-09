#Module algorithms

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../core/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")
add_library(cil_alg SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/procrustes.cpp"  "${CMAKE_CURRENT_SOURCE_DIR}/include/cil/algorithms/algorithms.h")


add_definitions(-D_CILAPI)
