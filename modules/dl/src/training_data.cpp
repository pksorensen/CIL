
#include "cil/dl/dl.h"


using namespace cil;
using namespace dl;

TrainingData::TrainingData()
{

}
TrainingData::~TrainingData()
{
	clear();
}
void TrainingData::clear()
{

}
int TrainingData::get_num_batches() const
{
	return num_samples / batch_size;
}