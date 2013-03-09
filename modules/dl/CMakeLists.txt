#Module DeepLearning

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../core/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")

set(DL_PUBLIC_HDR
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cil/dl/dl.h"
)
set(DL_SRC
	"${CMAKE_CURRENT_SOURCE_DIR}/src/nn.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/dl_params.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/training_data.cpp"
)

add_library(cil_dl SHARED  ${DL_SRC} ${DL_PUBLIC_HDR} )
set_target_properties(cil_dl PROPERTIES COMPILE_FLAGS "-D_CILAPI")
