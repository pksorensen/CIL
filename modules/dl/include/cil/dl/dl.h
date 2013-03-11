#ifndef __CIL_DL_H
#define __CIL_DL_H

#include "cil/core/core.h"
#include <cil/opencl/opencl.h>

#include "Eigen/eigen"
#include <vector>

namespace cil
{
#define CIL_DL_SIGMOID 'sigm';
#define CIL_DL_OPTIMAL_TANH 'tanh_opt';
namespace dl
{

struct CIL_EXPORTS DLParams
{
	bool			normalize_input;			//= 1;            %  normalize input elements to be between [-1 1]. Note: use a linear output function if training auto-encoders with normalized inputs
	char*			activation_function;		//= 'tanh_opt';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
	float			learning_rate;				//= 2;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
	float			momentum;					//= 0.5;          %  Momentum
	float			weight_penalty_L2;			//= 0;            %  L2 regularization
	float			non_sparsity_penalty;		//= 0;            %  Non sparsity penalty
	float			sparsity_target;			//= 0.05;         %  Sparsity target
	float			input_zero_masked_fraction;	//= 0;            %  Used for Denoising AutoEncoders
	float			dropoutFraction;			//= 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
//	nn.testing									//= 0;            %  Internal variable. nntest sets this to one.
	char*			output;						//= 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
	int				n;
	Eigen::VectorXi	architecture;

	DLParams();
	virtual ~DLParams();
	
	DLParams( Eigen::VectorXi & architecture );
	DLParams( int* architecture, int n);


};
#define GPU_MINIBATCHES_SWAP_CHAIN 2
class CIL_EXPORTS TrainingData
{
public:
	TrainingData(uint samples,uint attributes, uint epocs = 1, uint batch_size = 0 );
	virtual ~TrainingData();
	void clear();

	int get_num_batches() const;

	uint			batch_size;
	uint			num_epochs;
	uint			num_samples;
	uint			num_attributes;
	
	void create_minibatches();
	cl::CLMatrix* gpu_load_minibatch(cl_uint n);

	virtual const float* train_data() {return 0;};
private:
	cl::CLMatrix **	m_minibatch_swapchain;
	IndexVector m_idx;
};


class CIL_EXPORTS NeuralNetwork
{
public:
	NeuralNetwork(const DLParams & params=DLParams());

	NeuralNetwork(NeuralNetwork const&);				// Don't Implement
    void operator=(NeuralNetwork const&);				// Don't implement

	virtual ~NeuralNetwork();


	bool train(TrainingData &  train_data);
	bool set_params(const DLParams & params);

	std::vector<cl::CLMatrix *> & get_weights(){return m_weights;};
private:
	bool			initialize();


	std::vector<cl::CLMatrix *> m_weights;
	std::vector<cl::CLMatrix *> m_activations;

	DLParams		m_params;
	cl::CLManager & m_clmanager;
};






}//namespace dl
}//namespace cil

#endif