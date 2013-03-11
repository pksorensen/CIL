
#include "cil/dl/dl.h"
#include "cil/algorithms/algorithms.h"


using namespace cil;
using namespace dl;

TrainingData::TrainingData(uint samples,uint attributes, uint epocs, uint _batch_size)
	: num_samples(samples), num_epochs(epocs),
	num_attributes(attributes),
	batch_size(_batch_size==0?samples:_batch_size)
{
	m_minibatch_swapchain = new cl::CLMatrix*[GPU_MINIBATCHES_SWAP_CHAIN] ;
	for(int n=0;n<GPU_MINIBATCHES_SWAP_CHAIN;++n)
		m_minibatch_swapchain[n]=0;
}
TrainingData::~TrainingData()
{
	for(int n=0;n<GPU_MINIBATCHES_SWAP_CHAIN;++n)
		if(m_minibatch_swapchain[n]!=0) 
			cl::CLManager::getInstance().destroyMatrix(m_minibatch_swapchain[n]);

	delete[] m_minibatch_swapchain;
}
void TrainingData::clear()
{

}
int TrainingData::get_num_batches() const
{
	return num_samples / batch_size;
}
cl::CLMatrix* TrainingData::gpu_load_minibatch(cl_uint n)
{
	return m_minibatch_swapchain[n];
}
void TrainingData::create_minibatches()
{
	int err;
	m_idx.resize(num_samples);
	indexes(m_idx);
	alg::knuth_shuffle(m_idx);
	const uint* perm = m_idx.data();

	cl::CLManager& m = cl::CLManager::getInstance();
	cl_event* events = new cl_event[batch_size*GPU_MINIBATCHES_SWAP_CHAIN];
	size_t sample_mem_size = num_attributes*sizeof(float);

	for(int n = 0; n<GPU_MINIBATCHES_SWAP_CHAIN;++n)
	{
		m_minibatch_swapchain[n] = m.createMatrix(
			batch_size,num_attributes,CL_MEM_READ_ONLY); 
		
		for(int j =0;j < batch_size;++j)
		{
			
			const float* ptr = train_data()+num_attributes*perm[j];
			clEnqueueWriteBuffer(m.m_gpu_queue,m_minibatch_swapchain[n]->get_buffer(),
				CL_FALSE,j*sample_mem_size,sample_mem_size,ptr,0,NULL,events+(n*GPU_MINIBATCHES_SWAP_CHAIN)+j);

		}
		
		

	}
	clWaitForEvents(batch_size*GPU_MINIBATCHES_SWAP_CHAIN,events);
	delete[] events;

}
