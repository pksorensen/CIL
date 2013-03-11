

#include <cil/opencl/opencl.h>
#include <cil/dl/dl.h>
#include <Eigen\Dense>
#include <iostream>
using namespace cil;
using namespace cil::cl;
#include <ctime>


#define PRINT 0
int main()
{
	using namespace std;


	CLManager& manager = CLManager::getInstance();
	CLManager& manager1 = CLManager::getInstance();

	if (manager.status != 0)
		std::cout << manager.status << std::endl;

	int arc[4] = {784, 500, 500, 2000};
	dl::DLParams params(arc,4);


	dl::NeuralNetwork nn(params);


	
	CLMatrix* X1 = manager.createMatrix(784,500);
	CLMatrix* X2 = manager.createMatrix(500,5000);
	CLMatrix* X3 = manager.createMatrix(5000,2000);
	float* Z1 = new float[X1->numel()];
	float* Z2 = new float[X2->numel()];
	float* Z3 = new float[X3->numel()];
	Eigen::Map<Eigen::VectorXf> Y1(Z1,X1->numel());
	Eigen::Map<Eigen::VectorXf> Y2(Z2,X2->numel());
	Eigen::Map<Eigen::VectorXf> Y3(Z3,X3->numel());

	
	std::cout << "START" << std::endl;
	for(int n = 0; n< 100;++n)
	{
		clock_t begin = clock();

		manager.matrixRandomFill(X1,n);
		manager.matrixRandomFill(X2,n);
		manager.matrixRandomFill(X3,n);

		
		clEnqueueReadBuffer(manager.m_gpu_queue, X1->get_buffer(), CL_TRUE, 0, sizeof(float)*X1->numel(), Z1, 0, NULL, NULL);
		clEnqueueReadBuffer(manager.m_gpu_queue, X2->get_buffer(), CL_TRUE, 0, sizeof(float)*X2->numel(), Z2, 0, NULL, NULL);
		clEnqueueReadBuffer(manager.m_gpu_queue, X3->get_buffer(), CL_TRUE, 0, sizeof(float)*X3->numel(), Z3, 0, NULL, NULL);

#if PRINT		
		std::cout << (Y1.array()>0 && Y1.array()<0.2).count() << " "
			<< (Y1.array()>0.2 && Y1.array()<0.4).count() << " "
			<< (Y1.array()>0.4 && Y1.array()<0.6).count() << " "
			<< (Y1.array()>0.6 && Y1.array()<0.8).count() << " "
			<< (Y1.array()>0.8 && Y1.array()<1).count() << " ";

		Y1 = Y1.array() - Y1.mean();
		Y2 = Y2.array() - Y2.mean();
		Y3 = Y3.array() - Y3.mean();

		std::cout << Y1.mean() << " " << sqrt((Y1.cwiseProduct(Y1)).sum() / (Y1.rows()-1)) << std::endl;
		std::cout << Y2.mean() << " " << sqrt(Y2.cwiseProduct(Y2).sum() / (Y2.rows()-1)) << std::endl;
		std::cout << Y3.mean() << " " << sqrt(Y3.cwiseProduct(Y3).sum() / (Y3.rows()-1)) << std::endl<<std::endl;
#endif
		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << elapsed_secs*1000 << std::endl;

	}
	//std::cout << "END" << std::endl;
	//	std::cout << Y1 << std::endl;

	return 0;
}