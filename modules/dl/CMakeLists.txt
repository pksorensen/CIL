#Module DeepLearning

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../core/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../opencl/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../algorithms/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")
include_directories(${OPENCL_INCLUDE_DIRS})

set(DL_PUBLIC_HDR
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cil/dl/dl.h"
)
set(DL_SRC
	"${CMAKE_CURRENT_SOURCE_DIR}/src/nn.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/dl_params.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/training_data.cpp"
)

add_library(cil_dl SHARED  ${DL_SRC} ${DL_PUBLIC_HDR} )
TARGET_LINK_LIBRARIES(cil_dl cil_cl )
set_target_properties(cil_dl PROPERTIES COMPILE_FLAGS "-D_CILAPI")
