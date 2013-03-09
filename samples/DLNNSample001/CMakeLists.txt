
include_directories("${CIL_MODULES_PATH}/core/include/")
include_directories("${CIL_MODULES_PATH}/dl/include/")
include_directories("${CIL_MODULES_PATH}/opencl/include/")
include_directories(${OPENCL_INCLUDE_DIRS})
#link_directories(${OPENCL_LIBRARIES})


add_executable(dl_nn_sample001 "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

 
TARGET_LINK_LIBRARIES(dl_nn_sample001 cil_dl cil_cl ${OPENCL_LIBRARIES})