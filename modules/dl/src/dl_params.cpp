#include "cil/dl/dl.h"

using namespace cil;
using namespace dl;


DLParams::DLParams( int* _architecture , int n)
	: n(n), normalize_input(1), activation_function(CIL_DL_OPTIMAL_TANH),
	learning_rate(2.0f), momentum(0.5f), weight_penalty_L2(0.0f),
	non_sparsity_penalty(0.0f),sparsity_target(0.05f),
	input_zero_masked_fraction(0.0f), dropoutFraction(0),
	output(CIL_DL_SIGMOID)
{
	if(_architecture!=0)
		this->architecture = Eigen::Map<Eigen::VectorXi>(_architecture,n);

}
DLParams::~DLParams()
{

}

//DLParams::DLParams(Eigen::VectorXi & arc)
//{
//	this->architecture = arc;
//	this->n = n;
//}