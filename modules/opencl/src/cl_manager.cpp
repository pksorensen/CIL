
#include "cil\opencl\opencl.h"
#include "CL\cl.h"


using namespace cil;
using namespace cl;




char* file2string(const char* filename, const char* preamble, size_t* final_length) {
        FILE * file_stream = NULL;
        size_t source_length;

        // open the OpenCL source code file
  file_stream = fopen(filename, "rb");
  if(!file_stream) return NULL;
        //if(fopen_s(&file_stream, filename, "rb") != 0) return NULL;

        size_t preamble_length = strlen(preamble);

        // get the length of the source code
        fseek(file_stream, 0, SEEK_END); 
        source_length = ftell(file_stream);
        fseek(file_stream, 0, SEEK_SET); 

        // allocate a buffer for the source code string and read it in
        char* source_str = (char *)malloc(source_length + preamble_length + 1); 
        memcpy(source_str, preamble, preamble_length);
        if (fread((source_str) + preamble_length, source_length, 1, file_stream) != 1) {
                fclose(file_stream);
                free(source_str);
                return 0;
        }

        // close the file and return the total length of the combined (preamble + source) string
        fclose(file_stream);
        if(final_length != 0) 
                *final_length = source_length + preamble_length;
        source_str[source_length + preamble_length] = '\0';

        return source_str;
};


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
		file2string("C:/Development/CIL/modules/opencl/kernels/matrix_random_fill.cl","",&length[0]),
		file2string("C:/Development/CIL/modules/opencl/kernels/matrix_matrix_multiplication.cl","",&length[1])
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
                  "-IC:/Development/CIL/external/Random123-1.07/include -D ACTIVATION_FUNCTION(X)=(1.7159f*tanh(2.0f/3.0f*X))",   // Compiler options, see the specifications for more details
                 0, //void (*pfn_notify)(cl_program, void *user_data), 
                 0);

	if ( err != CL_SUCCESS )
		ERROR(err, CLErrString(err));

	m_random_fill_kernel= clCreateKernel (p,   // The program where the kernel is
         "matrix_random_fill",   // The name of the kernel, i.e. the name of the kernel function as it's declared in the code
         &err);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));
		
	m_matrix_multiplication_kernel= clCreateKernel (p,   // The program where the kernel is
         "matrix_matrix_mul",   // The name of the kernel, i.e. the name of the kernel function as it's declared in the code
         &err);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));


	error = false;
	__END__


	return error;
}
CLMatrix* CLManager::m_m_mul(CLMatrix* mx,CLMatrix* my,CLMatrix* result)
{
	cl_int err = true;
	
	CLMatrix* res = result!=0? result : createMatrix(mx->m_rows, my->m_cols);
	assert(res!=0);

	FUNCNAME( "CLManager::initialize" );

	__BEGIN__	

	err |= clSetKernelArg ( m_matrix_multiplication_kernel,0, sizeof(cl_mem),&res->get_buffer());
	err |= clSetKernelArg ( m_matrix_multiplication_kernel,1, sizeof(cl_mem),&mx->get_buffer());
	err |= clSetKernelArg ( m_matrix_multiplication_kernel,2, sizeof(cl_mem),&my->get_buffer());
	err |= clSetKernelArg ( m_matrix_multiplication_kernel,3, sizeof(cl_uint),&mx->m_cols);
	err |= clSetKernelArg ( m_matrix_multiplication_kernel,4, sizeof(cl_uint),&my->m_cols);
	err |= clSetKernelArg ( m_matrix_multiplication_kernel,5, sizeof(cl_uint),&mx->m_rows);

	cl_uint local_ws[] = {16,16};
	while(local_ws[0] >  my->m_cols)
		local_ws[0] /=2;
	local_ws[1] = 256 / local_ws[0];

	while(local_ws[1] >  mx->m_rows)
		local_ws[1] /=2;
	
	while(local_ws[0]*2 <  my->m_cols && local_ws[0]*local_ws[1]<256)
		local_ws[0] *=2;
	

	cl_uint global_ws[] = {local_ws[0],local_ws[1]};
	while(global_ws[0] < my->m_cols)
		global_ws[0] += local_ws[0];
	while(global_ws[1] < mx->m_rows)
		global_ws[1] += local_ws[1];

	err = clEnqueueNDRangeKernel(m_gpu_queue, m_random_fill_kernel, 2, NULL, global_ws, local_ws, 0, NULL, NULL);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));
	
	err = CL_SUCCESS;
	__END__

	if (err != CL_SUCCESS && !result)
	{
		destroyMatrix(res);
	}

	return res;
}
int CLManager::matrixRandomFill(CLMatrix* mx, cl_uint seed )
{
	cl_uint size = mx->numel();
	cl_mem buf = mx->get_buffer();
	cl_int err;
	int error = true;
	
	FUNCNAME( "CLManager::initialize" );

	__BEGIN__

	err |= clSetKernelArg ( m_random_fill_kernel,0,	sizeof(cl_uint),&seed);
	err |= clSetKernelArg ( m_random_fill_kernel,1, sizeof(cl_mem),	&buf);
	err |= clSetKernelArg ( m_random_fill_kernel,2,	sizeof(cl_uint),&size);

	// Launching kernel
	size = size/4+(size%4!=0);
	const size_t local_ws = 256;	// Number of work-items per work-group
// shrRoundUp returns the smallest multiple of local_ws bigger than size
	const size_t global_ws = ((size) / local_ws)*local_ws + (size%local_ws!=0)*local_ws;	// Total number of work-items
	err = clEnqueueNDRangeKernel(m_gpu_queue, m_random_fill_kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));

//	float* check = new float[mx->numel()];
//	clEnqueueReadBuffer(m_gpu_queue, buf, CL_TRUE, 0, sizeof(float)*mx->numel(), check, 0, NULL, NULL);

	error = false;
	__END__

	return error;
}
CLMatrix* CLManager::createMatrix(cl_uint m, cl_uint n,cl_uint access)
{
	int error = true;
	CLMatrix* mx = new CLMatrix(this,m,n);

	FUNCNAME( "CLManager::createMatrix" );

	__BEGIN__
	
		
	mx->set_buffer(clCreateBuffer( m_gpu_context,
		access, n*m* sizeof(cl_float), NULL, &error ));
	if ( error != CL_SUCCESS )
			ERROR(error, CLErrString(error));


	mx->status = error;
	__END__
	
	if (error != CL_SUCCESS)
	{
		delete mx;
		mx=0;
	}
	else
		m_buffers.push_back(mx);
	return mx;
	
}
void CLManager::destroyMatrix(CLMatrix* mx)
{

	m_buffers.erase(std::remove(m_buffers.begin(), m_buffers.end(), mx), m_buffers.end());
	delete mx;
}
