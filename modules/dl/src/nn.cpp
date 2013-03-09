#include "cil\dl\dl.h"


using namespace cil;
using namespace dl;
using namespace Eigen;

NeuralNetwork::NeuralNetwork(const DLParams & params)
{
	set_params(params);
};
NeuralNetwork::~NeuralNetwork()
{

}
bool NeuralNetwork::set_params(const DLParams & params)
{
	bool ok = false;

	{
		m_params = params;


	ok = true;
	}

	return ok;
}
bool NeuralNetwork::train(const TrainingData & train_data)
{
	bool ok = false;
	int n_batches, batch_size, n_epochs, n_samples;
	int n, i;


	{
	
	batch_size	= train_data.batch_size;
	n_epochs	= train_data.num_epochs;
	n_samples	= train_data.num_samples;
	n_batches	= n_samples/batch_size;

	IndexVector idx(n_samples);
	indexes(idx);
	
	Eigen::VectorXf L(n_epochs*n_batches);
	n = 0;
	for(i=0;i<n_epochs;++i)
	{
		
		

	}
	


	ok = true;
	}

	return ok;
}
bool NeuralNetwork::initialize()
{
	bool ok = false;


	{




	ok = true;
	}

	return ok;
}