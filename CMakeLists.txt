cmake_minimum_required (VERSION 2.6)
project (CIL)



# The version number.
set (SIMLIB_VERSION_MAJOR 1)
set (SIMLIB_VERSION_MINOR 0)


set(EXTERNAL_PATH ${PROJECT_SOURCE_DIR}/external/)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

# Save libs and executables in the same place
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications" )


# ---------- Setup output Directories -------------------------
SET (CMAKE_LIBRARY_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/bin
   CACHE PATH
   "Single Directory for all Libraries"
   )

# --------- Setup the Executable output Directory -------------
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/bin
   CACHE PATH
   "Single Directory for all Executables."
   )

# --------- Setup the Executable output Directory -------------
SET (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/bin
   CACHE PATH
   "Single Directory for all static libraries."
   )

find_package(OpenCL REQUIRED)

set(EXTERNAL_PATH ${PROJECT_SOURCE_DIR}/external/)
set(RANDOM123_INCLUDE_DIRS "${EXTERNAL_PATH}Random123-1.07/include")
   
SUBDIRS(modules samples)
