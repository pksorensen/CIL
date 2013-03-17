#ifndef __CIL_OPENCL_H
#define __CIL_OPENCL_H

#include "cil/core/core.h"
#include <CL/opencl.h>
#include <vector>
#include <time.h>

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
	  { CL_INVALID_CONTEXT , "CL_INVALID_CONTEXT", },
	{ CL_INVALID_QUEUE_PROPERTIES, "CL_INVALID_QUEUE_PROPERTIES", },
	{ CL_INVALID_COMMAND_QUEUE , "CL_INVALID_COMMAND_QUEUE", },
	{ CL_INVALID_HOST_PTR , "CL_INVALID_HOST_PTR", },
	{ CL_INVALID_MEM_OBJECT , "CL_INVALID_MEM_OBJECT", },
	{ CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", },
	{ CL_INVALID_IMAGE_SIZE , "CL_INVALID_IMAGE_SIZE", },
	{ CL_INVALID_SAMPLER, "CL_INVALID_SAMPLER", },
	{ CL_INVALID_BINARY , "CL_INVALID_BINARY", },
	{ CL_INVALID_BUILD_OPTIONS , "", },
	{ CL_INVALID_PROGRAM, "", },
	{ CL_INVALID_PROGRAM_EXECUTABLE , "", },
	{ CL_INVALID_KERNEL_NAME , "", },
	{ CL_INVALID_KERNEL_DEFINITION , "CL_INVALID_KERNEL_DEFINITION", },
	{ CL_INVALID_KERNEL, "", },
	{ CL_INVALID_ARG_INDEX , "", },
	{ CL_INVALID_ARG_VALUE , "", },
	{ CL_INVALID_ARG_SIZE , "", },
	{ CL_INVALID_KERNEL_ARGS , "", },
	{ CL_INVALID_WORK_DIMENSION , "", },
	{ CL_INVALID_WORK_GROUP_SIZE , "", },
	{ CL_INVALID_WORK_ITEM_SIZE, "", },
	{ CL_INVALID_GLOBAL_OFFSET , "", },
	{ CL_INVALID_EVENT_WAIT_LIST, "", },
	{ CL_INVALID_EVENT , "", },
	{ CL_INVALID_OPERATION , "", },
	{ CL_INVALID_GL_OBJECT , "", },
	{ CL_INVALID_BUFFER_SIZE , "", },
	{ CL_INVALID_MIP_LEVEL , "", },
	{ CL_INVALID_GLOBAL_WORK_SIZE , "", },
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


class CLMatrix;
class CIL_EXPORTS CLManager
{
public:
	
	static  CLManager& getInstance();
	static int count;

	virtual ~CLManager();

	bool	initialize(const char* platform_filter=0);
	void	cleanup();

	CLMatrix* createMatrix(cl_uint n, cl_uint m,float* data =0, cl_uint access = CL_MEM_READ_WRITE );
	
	void destroyMatrix(CLMatrix* mx);

	int matrixRandomFill(CLMatrix* mx, cl_uint seed = rand());
	int load_gpu_data(CLMatrix* mx,float*data);
	
	int status;
	time_t					m_seed;

public:
	int load_mm_mult_kernel();
	int isloaded_mm_mult_kernel() const {return m_kernel_mm_mult!=0;};
	int matrix_matrix_multiplication(CLMatrix* A, CLMatrix* B, CLMatrix*C);
private:
	cl_kernel		m_kernel_mm_mult;

private:
	CLManager() : 
		m_platforms_list(0), status(0),
		m_kernel_mm_mult(0), m_last_calc(0)
	{
		status = initialize();
		m_seed = time(NULL);
	};
	
	CLManager(CLManager const&);				// Don't Implement
    void operator=(CLManager const&);			// Don't implement

	
	
	std::vector<CLMatrix*> m_buffers;
public:
	cl_event			m_last_calc;
	// OpenCL specific
	cl_platform_id *	m_platforms_list;
	cl_uint				m_num_platforms;

	cl_device_id		m_gpu_device_id;	
	cl_context			m_gpu_context;
	cl_command_queue	m_gpu_queue;
	cl_command_queue	m_gpu_queue_loader;


	cl_device_id		m_cpu_device_id;	
	cl_context			m_cpu_context;
	cl_command_queue	m_cpu_queue;


	cl_kernel			m_kernel_random_fill;
	cl_kernel			m_kernel_nnfeedforward;

};
class CIL_EXPORTS CLMatrix
{
public:
	CLMatrix(CLManager * manager, cl_uint m=0, cl_uint n=0);
	virtual ~CLMatrix();
	void set_buffer(cl_mem buf){m_buffer =buf;};
	const cl_mem & get_buffer() const {return m_buffer;} ;
	const cl_uint numel() const {return m_rows*m_cols;};


	Eigen::MatrixXf load_from_gpu()
	{
		Eigen::Matrix<float,-1,-1,Eigen::RowMajor> mat(m_rows,m_cols);
		clEnqueueReadBuffer(m_manager->m_gpu_queue_loader, m_buffer, CL_TRUE, 0, sizeof(float)*numel(), mat.data(), 0, NULL, NULL);	
		return mat;	
	}

	int			status;
	cl_uint		m_rows;
	cl_uint		m_cols;
private:
	CLManager * m_manager;
	cl_mem		m_buffer;

	
};




}//opencl
}//cil

#endif