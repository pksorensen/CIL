#include "cil\dl\dl.h"


using namespace cil;
using namespace cl;
using namespace dl;
using namespace Eigen;

NeuralNetwork::NeuralNetwork(const DLParams & params)
	: m_clmanager(cl::CLManager::getInstance()),
	m_feedforward_activation(0), m_feedforward_output(0)
{
	set_params(params);
};
NeuralNetwork::~NeuralNetwork()
{
	for(int n = 0; n<m_weights.size();++n)
		m_clmanager.destroyMatrix(m_weights[n]);
	for(int n = 0; n<m_activations.size();++n)
		m_clmanager.destroyMatrix(m_activations[n]);

	m_weights.clear();
}
void build_feedforward_kernel(const char** sourse, size_t* length, int code, cl_kernel& kernel,int* err,int output =0 )
{
	char str[255] = "-D";
	if (code == CIL_DL_OPTIMAL_TANH)
		std::strcat(str,CIL_DL_OPTIMAL_TANH_STR);
	else if(code == CIL_DL_SIGMOID)
		std::strcat(str,CIL_DL_SIGMOID_STR);
	else if(code == CIL_DL_LINEAR || code == CIL_DL_SOFTMAX)
		std::strcat(str,CIL_DL_LINEAR_STR);

	if(output)
		std::strcat(str," -DOUTPUT");

	cl_program p= clCreateProgramWithSource( cl::CLManager::getInstance().m_gpu_context,
                  1,   // number of files
                  sourse,   // array of strings, each one is a file
                  length,   // array specifying the file lengths
                  err);   // error code to be returned
	if (*err)
		return;


	*err = clBuildProgram (p, 
                  1,
				  &cl::CLManager::getInstance().m_gpu_device_id,
                  str,   // Compiler options, see the specifications for more details
                 0, //void (*pfn_notify)(cl_program, void *user_data), 
                 0);
	if (*err)
		return;

	kernel= clCreateKernel (p,   // The program where the kernel is
         "nn_feedforward",   // The name of the kernel, i.e. the name of the kernel function as it's declared in the code
         err);
	
};

CLMatrix* nn_kernel_feedforward(cl_kernel k, CLMatrix* mx,CLMatrix* my,CLMatrix* result, 
								CLMatrix* target = 0, CLMatrix* error = 0, CLMatrix** lserror = 0)
{
	cl_int err = true;
	CLManager & manager = CLManager::getInstance();

	CLMatrix* res = result!=0? result : manager.createMatrix(mx->m_rows, my->m_cols);
	assert(res!=0);

	FUNCNAME( "NN::nn_feedforward" );

	__BEGIN__	

	err |= clSetKernelArg ( k,0, sizeof(cl_mem),&res->get_buffer());
	err |= clSetKernelArg ( k,1, sizeof(cl_mem),&mx->get_buffer());
	err |= clSetKernelArg ( k,2, sizeof(cl_mem),&my->get_buffer());
	err |= clSetKernelArg ( k,3, sizeof(cl_uint),&mx->m_cols);
	err |= clSetKernelArg ( k,4, sizeof(cl_uint),&my->m_cols);
	err |= clSetKernelArg ( k,5, sizeof(cl_uint),&mx->m_rows);
	


	size_t local_ws[] = {16,16};
	while(local_ws[0] >  my->m_cols)
		local_ws[0] /=2;
	local_ws[1] = 256 / local_ws[0];

	while(local_ws[1] >  mx->m_rows)
		local_ws[1] /=2;
	
	while(local_ws[0]*2 <  my->m_cols && local_ws[0]*local_ws[1]<256)
		local_ws[0] *=2;
	

	size_t global_ws[] = {local_ws[0],local_ws[1]};
	while(global_ws[0] < my->m_cols)
		global_ws[0] += local_ws[0];
	while(global_ws[1] < mx->m_rows)
		global_ws[1] += local_ws[1];


	if(target)
	{
		err |= clSetKernelArg ( k,6, sizeof(cl_mem),&target->get_buffer());
		err |= clSetKernelArg ( k,7, sizeof(cl_mem),&error->get_buffer());
		err |= clSetKernelArg ( k,8, sizeof(float)*local_ws[0]*local_ws[1],0);
		if(!*lserror)
			*lserror = manager.createMatrix(1,global_ws[0]*global_ws[1] / (local_ws[0]*local_ws[1]));
		err |= clSetKernelArg ( k,9, sizeof(cl_mem),&(*lserror)->get_buffer());
	}

	err = clEnqueueNDRangeKernel(manager.m_gpu_queue, k, 2, NULL, global_ws, local_ws, 0, NULL, NULL);
	if ( err != CL_SUCCESS )
			ERROR(err, CLErrString(err));
	
	err = CL_SUCCESS;
	__END__

	if (err != CL_SUCCESS && !result)
	{
		manager.destroyMatrix(res);
	}

	return res;
}

bool NeuralNetwork::set_params(const DLParams & params)
{
	int err = true;
	int * arc;

	FUNCNAME( "NeuralNetwork::set_params" );

	__BEGIN__

	cl::CLManager& manager = cl::CLManager::getInstance();

	m_params = params;
	arc = m_params.architecture.data();

	for(int n = 1; n<m_params.n; ++n)
	{
		m_weights.push_back(manager.createMatrix(arc[n-1]+1,arc[n]));
		manager.matrixRandomFill(m_weights[n-1]);

	}
	size_t length[3] = {0};
	const char * sourceStr[] = {
		file2string("C:/Development/CIL/modules/opencl/kernels/nn_feedforward.cl","",&length[0])
	};

	build_feedforward_kernel(sourceStr,length,params.output, m_feedforward_output , &err,1);
	if ( err != CL_SUCCESS )
		ERROR(err, cil::cl::CLErrString(err));
	build_feedforward_kernel(sourceStr,length,params.activation_function, m_feedforward_activation, &err);
	if ( err != CL_SUCCESS )
		ERROR(err, cil::cl::CLErrString(err));





	err = false;
	__END__


	return error;
};
bool NeuralNetwork::train( TrainingData  & train_data)
{
	bool err = true;
	int n_batches, batch_size, n_epochs, n_samples;
	int n, i;

	FUNCNAME( "NeuralNetwork::train" );

	__BEGIN__
	
	train_data.create_minibatches();	
	batch_size	= train_data.batch_size;
	n_epochs	= train_data.num_epochs;
	n_samples	= train_data.num_samples;
	n_batches	= n_samples/batch_size;
	
	cl::CLManager& manager = cl::CLManager::getInstance();
	for(n = 1; n<m_params.n; ++n)
		m_activations.push_back(manager.createMatrix(batch_size,m_params.architecture[n]));

	m_error = manager.createMatrix(batch_size,m_params.architecture[--n]);
	//m_l		= manager.createMatrix(1,n_epochs);
	CLMatrix* m_l = 0;


	
	Eigen::VectorXf L(n_epochs*n_batches);
	n = 0;

	for(i=0;i<n_epochs;++i)
	{

		
		for(int z = 1; z <= 1+0*n_batches;++z)
		{

			//Feed Forward;
			nn_kernel_feedforward(m_feedforward_activation,train_data.gpu_load_minibatch(z),m_weights[0],m_activations[0]);
			nn_kernel_feedforward(m_feedforward_activation,m_activations[0],m_weights[1],m_activations[1]);
			nn_kernel_feedforward(m_feedforward_activation,m_activations[1],m_weights[2],m_activations[2]);
			nn_kernel_feedforward(m_feedforward_output,m_activations[2],m_weights[3],m_activations[3],
				train_data.gpu_load_minibatch_targets(z),m_error,&m_l);
			if (m_params.output == CIL_DL_SOFTMAX)
				ERROR(-1,"NOT IMPLEMENTED");	
			clFinish(manager.m_gpu_queue);
				

			//backp

		}
		

	}
	


	err = false;
	__END__

	return err;
}
bool NeuralNetwork::initialize()
{
	bool ok = false;


	{




	ok = true;
	}

	return ok;
}