#Module OpenCL

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../core/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")
include_directories(${OPENCL_INCLUDE_DIRS})

set(CL_PUBLIC_HDR
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cil/opencl/opencl.h"
)
set(CL_SRC
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cl_manager.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cl_matrix.cpp"
)

add_library(cil_cl STATIC  ${CL_SRC} ${CL_PUBLIC_HDR} )
TARGET_LINK_LIBRARIES(cil_cl cil_core ${OPENCL_LIBRARIES})
set_target_properties(cil_cl PROPERTIES COMPILE_FLAGS "-D_CILAPI")
