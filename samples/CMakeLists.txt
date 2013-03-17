if(NOT CIL_SAMPLES_PATH)
  set(CIL_SAMPLES_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

file(GLOB all_valid_subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*/CMakeLists.txt")

foreach(dir ${all_valid_subdirs})
    message(STATUS "path = ${dir}")
    if(${dir} MATCHES "^([^/]*)//CMakeLists.txt")
        string(REGEX REPLACE "^([^/]*)//CMakeLists.txt" "\\1" dir_trimmed ${dir})
        add_subdirectory(${dir_trimmed})
    endif()
endforeach(dir)



add_executable(gpu_matrix_sample001 "${CMAKE_CURRENT_SOURCE_DIR}/gpu_matrix_sample001.cpp")
TARGET_LINK_LIBRARIES(gpu_matrix_sample001 cil_cl ${OPENCL_LIBRARIES})
set_target_properties(gpu_matrix_sample001 PROPERTIES COMPILE_FLAGS /I"${EXTERNAL_PATH}")
set_property(TARGET gpu_matrix_sample001 APPEND PROPERTY INCLUDE_DIRECTORIES "${CIL_MODULES_PATH}/core/include/")
set_property(TARGET gpu_matrix_sample001 APPEND PROPERTY INCLUDE_DIRECTORIES "${CIL_MODULES_PATH}/opencl/include/")
set_property(TARGET gpu_matrix_sample001 APPEND PROPERTY INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIRS}")
set_property(TARGET gpu_matrix_sample001 APPEND PROPERTY INCLUDE_DIRECTORIES "${RANDOM123_INCLUDE_DIRS}")
set_property(TARGET gpu_matrix_sample001 APPEND PROPERTY INCLUDE_DIRECTORIES "${CIL_MODULES_PATH}/dl/include/")


add_executable(eigen-with-viennacl "${CMAKE_CURRENT_SOURCE_DIR}/eigen-with-viennacl.cpp")
set_target_properties(eigen-with-viennacl PROPERTIES COMPILE_FLAGS /I"${EXTERNAL_PATH}")

add_executable(opencl_sample001 "${CMAKE_CURRENT_SOURCE_DIR}/opencl_sample001.cpp")
TARGET_LINK_LIBRARIES(opencl_sample001 cil_dl cil_cl ${OPENCL_LIBRARIES})
set_property(TARGET opencl_sample001 APPEND PROPERTY INCLUDE_DIRECTORIES "${CIL_MODULES_PATH}/core/include/")
set_property(TARGET opencl_sample001 APPEND PROPERTY INCLUDE_DIRECTORIES "${CIL_MODULES_PATH}/opencl/include/")
set_property(TARGET opencl_sample001 APPEND PROPERTY INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIRS}")


add_executable(gpu_matrix_randomnumbers "${CMAKE_CURRENT_SOURCE_DIR}/gpu_matrix_randomnumbers.cpp")
TARGET_LINK_LIBRARIES(gpu_matrix_randomnumbers cil_dl cil_cl ${OPENCL_LIBRARIES})
set_property(TARGET gpu_matrix_randomnumbers APPEND PROPERTY INCLUDE_DIRECTORIES "${CIL_MODULES_PATH}/core/include/")
set_property(TARGET gpu_matrix_randomnumbers APPEND PROPERTY INCLUDE_DIRECTORIES "${CIL_MODULES_PATH}/opencl/include/")
set_property(TARGET gpu_matrix_randomnumbers APPEND PROPERTY INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIRS}")
set_property(TARGET gpu_matrix_randomnumbers APPEND PROPERTY INCLUDE_DIRECTORIES "${RANDOM123_INCLUDE_DIRS}")
set_property(TARGET gpu_matrix_randomnumbers APPEND PROPERTY INCLUDE_DIRECTORIES "${CIL_MODULES_PATH}/dl/include/")



set(OPENCLRANDOMS
	"${EXTERNAL_PATH}/Random123-1.07/examples/my_pi_opencl.c"
	"${EXTERNAL_PATH}/Random123-1.07/examples/pi_check.h"
	"${EXTERNAL_PATH}/Random123-1.07/examples/util_opencl.h"
)
add_executable(opencl_sample002 ${OPENCLRANDOMS})

set_property(TARGET opencl_sample002 APPEND PROPERTY INCLUDE_DIRECTORIES "${RANDOM123_INCLUDE_DIRS}")
set_property(TARGET opencl_sample002 APPEND PROPERTY INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIRS}")