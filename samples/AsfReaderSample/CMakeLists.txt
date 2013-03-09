


include_directories("${CIL_MODULES_PATH}/core/include/")
include_directories("${CIL_MODULES_PATH}/algorithms/include/")
add_executable(asf_reader_sample "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp") 
TARGET_LINK_LIBRARIES(asf_reader_sample cil_alg)