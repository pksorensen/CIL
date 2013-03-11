#include "cil\dl\dl.h"


using namespace cil;
using namespace dl;
using namespace Eigen;

NeuralNetwork::NeuralNetwork(const DLParams & params)
	: m_clmanager(cl::CLManager::getInstance())
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
bool NeuralNetwork::set_params(const DLParams & params)
{
	bool error = true;
	int * arc;

	FUNCNAME( "NeuralNetwork::set_params" );

	__BEGIN__
	m_params = params;
	arc = m_params.architecture.data();

	for(int n = 1; n<m_params.n; ++n)
	{
		m_weights.push_back(cl::CLManager::getInstance()
			.createMatrix(arc[n-1],arc[n]));
		cl::CLManager::getInstance().matrixRandomFill(m_weights[n-1]);

	}

	


	error = false;
	__END__


	return error;
}
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
	{
		m_activations.push_back(manager.createMatrix(batch_size,m_params.architecture[n]));
		manager.matrixRandomFill(m_activations[n-1]);
	}

	
	Eigen::VectorXf L(n_epochs*n_batches);
	n = 0;
	for(i=0;i<n_epochs;++i)
	{

		
		for(int z = 0; z < 1+0*n_batches;++z)
		{

			//Feed Forward;
			manager.m_m_mul(train_data.gpu_load_minibatch(z),m_weights[0],m_activations[0]);
			manager.m_m_mul(m_activations[0],m_weights[1],m_activations[1]);
			manager.m_m_mul(m_activations[1],m_weights[2],m_activations[2]);
			manager.m_m_mul(m_activations[2],m_weights[3],m_activations[3]);
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