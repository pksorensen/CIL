#Module CORE

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")

set(CORE_PUBLIC_HDR
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cil/core/core.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cil/core/types_c.h"
)
set(CORE_SRC
	"${CMAKE_CURRENT_SOURCE_DIR}/src/system.cpp"
)

add_library(cil_core SHARED  ${CORE_SRC} ${CORE_PUBLIC_HDR} )
set_target_properties(cil_core PROPERTIES COMPILE_FLAGS "-D_CILAPI")
