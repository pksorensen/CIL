#ifndef __CIL_OPENCL_H
#define __CIL_OPENCL_H

#include "cil/core/core.h"
#include "CL/opencl.h"

namespace cil
{
namespace cl
{


/*
 * CLErrString --
 *
 *      Utility function that converts an OpenCL status into a human
 *      readable string.
 *
 * Results:
 *      const char * pointer to a static string.
 */

static const char *
CLErrString(cl_int status) {
   static struct { cl_int code; const char *msg; } error_table[] = {
      { CL_SUCCESS, "success" },
      { CL_DEVICE_NOT_FOUND, "device not found", },
      { CL_DEVICE_NOT_AVAILABLE, "device not available", },
      { CL_COMPILER_NOT_AVAILABLE, "compiler not available", },
      { CL_MEM_OBJECT_ALLOCATION_FAILURE, "mem object allocation failure", },
      { CL_OUT_OF_RESOURCES, "out of resources", },
      { CL_OUT_OF_HOST_MEMORY, "out of host memory", },
      { CL_PROFILING_INFO_NOT_AVAILABLE, "profiling not available", },
      { CL_MEM_COPY_OVERLAP, "memcopy overlaps", },
      { CL_IMAGE_FORMAT_MISMATCH, "image format mismatch", },
      { CL_IMAGE_FORMAT_NOT_SUPPORTED, "image format not supported", },
      { CL_BUILD_PROGRAM_FAILURE, "build program failed", },
      { CL_MAP_FAILURE, "map failed", },
      { CL_INVALID_VALUE, "invalid value", },
      { CL_INVALID_DEVICE_TYPE, "invalid device type", },
      { 0, NULL },
   };
   static char unknown[25];
   int ii;

   for (ii = 0; error_table[ii].msg != NULL; ii++) {
      if (error_table[ii].code == status) {
         return error_table[ii].msg;
      }
   }

   _snprintf(unknown, sizeof unknown, "unknown error %d", status);
   return unknown;
}



class CIL_EXPORTS CLManager
{
public:
	CLManager();
	virtual ~CLManager();


	bool	initialize(const char* platform_filter=0);
	void	cleanup();
private:

	// OpenCL specific
	cl_platform_id *	m_platforms_list;
	cl_uint				m_num_platforms;

	cl_context			m_gpu_context;
	cl_context			m_cpu_context;

};





}//opencl
}//cil

#endif