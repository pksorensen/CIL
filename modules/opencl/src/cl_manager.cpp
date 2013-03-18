
#include "cil\opencl\opencl.h"
#include "CL\cl.h"


using namespace cil;
using namespace cl;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#define BLOCK_SIZE_STR "-DBLOCK_SIZE=16"
#endif


int CLManager::count = 0;

CLManager::~CLManager()
{
	cleanup();
}
void CLManager::cleanup()
{
	if(m_platforms_list)
		delete m_platforms_list;
	m_platforms_list=0;

	clReleaseContext(m_cpu_context);
	clReleaseContext(m_gpu_context);
	clReleaseCommandQueue(m_gpu_queue);
	clReleaseCommandQueue(m_cpu_queue);
}
CLManager& CLManager::getInstance()
{
	static CLManager instance; // Guaranteed to be destroyed.
                                // Instantiated on first use.
    return instance;
}
bool CLManager::initialize( const char* platform_filter)
{
	 CLManager::count++;
	bool error = true;
	int err;                            // error code returned from api calls


    char pPlatformName[128] = { 0 };


	FUNCNAME( "CLManager::initialize" );

	__BEGIN__


	err = clGetPlatformIDs(0, m_platforms_list, &m_num_platforms);            
	if(err != CL_SUCCESS || m_num_platforms==0 )
		ERROR(err,CLErrString(err));
	
	m_platforms_list = (cl_platform_id *) malloc(sizeof(cl_platform_id) * m_num_platforms);
    if ((err = clGetPlatformIDs(m_num_platforms, m_platforms_list, NULL)) != CL_SUCCESS) {
		ERROR(err, CLErrString(err));
    }
	
	if ( platform_filter )
	for (cl_uint ui = 0; ui < m_num_platforms; ++ui)
    {
        err = clGetPlatformInfo(m_platforms_list[ui], CL_PLATFORM_NAME, 128 * sizeof(char), pPlatformName, NULL);
        if (err == CL_SUCCESS && !strcmp(pPlatformName, platform_filter)) //"Intel(R) OpenCL"
		{
			cl_platform_id temp = m_platforms_list[0];
			m_platforms_list[0] = m_platforms_list[ui];
			m_platforms_list[ui] = temp;

		}

    }

	// Connect to a GPU compute device
    //
    err = clGetDeviceIDs(m_platforms_list[0], CL_DEVICE_TYPE_GPU , 1, &m_gpu_device_id, NULL);
    if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));

	
	// Connect to a CPU compute device
    //
	err = clGetDeviceIDs(m_platforms_list[0], CL_DEVICE_TYPE_CPU , 1, &m_cpu_device_id, NULL);
    if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));
	
	// Create a compute context 
    //
	m_gpu_context = clCreateContext(0, 1, &m_gpu_device_id, NULL, NULL, &err);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));
	m_cpu_context = clCreateContext(0, 1, &m_cpu_device_id, NULL, NULL, &err);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));
	
	

	m_gpu_queue = clCreateCommandQueue( m_gpu_context,
		m_gpu_device_id, 0, &err );
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));

	m_gpu_queue_loader = clCreateCommandQueue( m_gpu_context,
		m_gpu_device_id, 0, &err );
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));


	m_cpu_queue = clCreateCommandQueue( m_cpu_context,
		m_cpu_device_id, 0, &err );
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));

	size_t length[3] = {0};
	const char * sourceStr[] = {
		file2string("D:/GitHub/CIL/modules/opencl/kernels/matrix_random_fill.cl","",&length[0]),
		file2string("D:/GitHub/CIL/modules/opencl/kernels/nn_feedforward.cl","",&length[1])
							};

	
	cl_program p= clCreateProgramWithSource( m_gpu_context,
                  2,   // number of files
                  sourceStr,   // array of strings, each one is a file
                  (size_t*)length,   // array specifying the file lengths
                  &err);   // error code to be returned
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));
	
	cl_int err = clBuildProgram (p, 
                  1,
				  &m_gpu_device_id,
                  "-ID:/GitHub/CIL/external/Random123-1.07/include -D ACTIVATION_FUNCTION(X)=(1.7159f*tanh(2.0f/3.0f*X))",   // Compiler options, see the specifications for more details
                 0, //void (*pfn_notify)(cl_program, void *user_data), 
                 0);
	if ( err != CL_SUCCESS )
		ERROR(err, CLErrString(err));

	m_kernel_random_fill= clCreateKernel (p,   // The program where the kernel is
         "matrix_random_fill",   // The name of the kernel, i.e. the name of the kernel function as it's declared in the code
         &err);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));
		
	m_kernel_nnfeedforward= clCreateKernel (p,   // The program where the kernel is
         "nn_feedforward",   // The name of the kernel, i.e. the name of the kernel function as it's declared in the code
         &err);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));

	cl_program


	error = false;
	__END__


	return error;
}

int CLManager::matrixRandomFill(CLMatrix* mx, cl_uint seed )
{
	cl_uint size = mx->numel();
	cl_mem buf = mx->get_buffer();
	cl_int err;
	int error = true;
	
	FUNCNAME( "CLManager::initialize" );

	__BEGIN__

	err = clSetKernelArg ( m_kernel_random_fill,0,	sizeof(cl_uint),&seed);
	err |= clSetKernelArg ( m_kernel_random_fill,1, sizeof(cl_mem),	&buf);
	err |= clSetKernelArg ( m_kernel_random_fill,2,	sizeof(cl_uint),&size);

	// Launching kernel
	size = size/4+(size%4!=0);
	const size_t local_ws = 256;	// Number of work-items per work-group
// shrRoundUp returns the smallest multiple of local_ws bigger than size
	const size_t global_ws = ((size) / local_ws)*local_ws + (size%local_ws!=0)*local_ws;	// Total number of work-items
	err = clEnqueueNDRangeKernel(m_gpu_queue, m_kernel_random_fill, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));

//	float* check = new float[mx->numel()];
//	clEnqueueReadBuffer(m_gpu_queue, buf, CL_TRUE, 0, sizeof(float)*mx->numel(), check, 0, NULL, NULL);

	error = false;
	__END__

	return error;
}
CLMatrix* CLManager::createMatrix(cl_uint m, cl_uint n, float* data,cl_uint access )
{
	int error = true;
	CLMatrix* mx = new CLMatrix(this,m,n);

	FUNCNAME( "CLManager::createMatrix" );

	__BEGIN__
	
		
	mx->set_buffer(clCreateBuffer( m_gpu_context,
		access, n*m* sizeof(cl_float), NULL, &error ));
	if ( error != CL_SUCCESS )
			ERROR(error, CLErrString(error));
	
	if(data!=0){
		error = clEnqueueWriteBuffer(m_gpu_queue_loader,mx->get_buffer(),CL_TRUE,
			0,sizeof(float)*m*n,data,0,0,0);
		if ( error != CL_SUCCESS )
			ERROR(error, CLErrString(error));
	}

	mx->status = error;
	__END__
	
	
	if (error != CL_SUCCESS)
	{
		delete mx;
		mx=0;
	}else
		m_buffers.push_back(mx);

	return mx;
	
}
void CLManager::destroyMatrix(CLMatrix* mx)
{

	clReleaseMemObject(mx->get_buffer());
	m_buffers.erase(std::remove(m_buffers.begin(), m_buffers.end(), mx), m_buffers.end());
	delete mx;
}


int CLManager::load_mm_mult_kernel()
{
	int err = true;
	FUNCNAME( "CLManager::load_mm_mult_kernel" );


	__BEGIN__

	size_t length[1] = {0};
	const char * sourceStr[] = {
		file2string("D:/GitHub/CIL/modules/opencl/kernels/m_m_mul2.cl","",&length[0])
	};
		
	cl_program p= clCreateProgramWithSource( m_gpu_context,
                  1,   // number of files
                  sourceStr,   // array of strings, each one is a file
                  (size_t*)length,   // array specifying the file lengths
                  &err);   // error code to be returned
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));
	
	cl_int err = clBuildProgram (p, 
                  1,
				  &m_gpu_device_id,
                 BLOCK_SIZE_STR,
                 0, //void (*pfn_notify)(cl_program, void *user_data), 
                 0);
	if ( err != CL_SUCCESS )
		ERROR(err, CLErrString(err));
	
	m_kernel_mm_mult= clCreateKernel (p,   // The program where the kernel is
         "m_m_mul",   // The name of the kernel, i.e. the name of the kernel function as it's declared in the code
         &err);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));

		err = false;
	__END__

	return err;
}

int CLManager::matrix_matrix_multiplication(CLMatrix* A, CLMatrix* B, CLMatrix*C)
{
	int err = true;
	FUNCNAME( "CLManager::matrix_matrix_multiplication" );


	__BEGIN__
	
	if (!isloaded_mm_mult_kernel())
		load_mm_mult_kernel();
	
	int block_size = BLOCK_SIZE;
	size_t local_ws[] = {block_size,block_size};
	size_t global_ws[] = {block_size,block_size};
	while(global_ws[0] < B->m_cols)
		global_ws[0] += local_ws[0];
	while(global_ws[1] < A->m_rows)
		global_ws[1] += local_ws[1];
	int lw = global_ws[1];
	err = clSetKernelArg ( m_kernel_mm_mult,0,	sizeof(cl_mem),&A->get_buffer());
	err |= clSetKernelArg ( m_kernel_mm_mult,1, sizeof(cl_mem),&B->get_buffer());
	err |= clSetKernelArg ( m_kernel_mm_mult,2,	sizeof(cl_mem),&C->get_buffer());
	//err |= clSetKernelArg ( m_kernel_mm_mult,3,	sizeof(float)*block_size*block_size,0);
	//err |= clSetKernelArg ( m_kernel_mm_mult,4,	sizeof(float)*block_size*block_size,0);
	err |= clSetKernelArg ( m_kernel_mm_mult,3,	sizeof(cl_int),&A->m_cols);
	err |= clSetKernelArg ( m_kernel_mm_mult,4,	sizeof(cl_int),&B->m_cols);
	err |= clSetKernelArg ( m_kernel_mm_mult,5,	sizeof(cl_int), &lw);
	
	//err |= clSetKernelArg ( m_kernel_mm_mult,3,	sizeof(float)*A->m_cols,NULL);	
	//err |= clSetKernelArg ( m_kernel_mm_mult,3,	sizeof(cl_uint),&A->m_cols);

	if ( err != CL_SUCCESS )
		ERROR(err, CLErrString(err));
	
	//size_t local_ws[] = {1,256};
	//while (local_ws[1]>A->m_rows)
	//	local_ws[1] /= 2;

	//size_t global_ws[] = {B->m_cols,local_ws[1]};
	//while(global_ws[1] < A->m_rows)
	//	global_ws[1]+=local_ws[1];
	


	err = clEnqueueNDRangeKernel(m_gpu_queue, m_kernel_mm_mult, 2, NULL, global_ws, local_ws, 0, NULL, 0);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));

//	float* check = new float[mx->numel()];
//	clEnqueueReadBuffer(m_gpu_queue, buf, CL_TRUE, 0, sizeof(float)*mx->numel(), check, 0, NULL, NULL);
	//clFinish(m_gpu_queue);

	err = false;
	__END__

	return err;
}
int CLManager::load_gpu_data(CLMatrix* mx,float*data)
{
	return clEnqueueReadBuffer(m_gpu_queue,mx->get_buffer(),CL_TRUE,0,
		sizeof(float)*mx->m_rows*mx->m_cols,data,0,0,0);

}