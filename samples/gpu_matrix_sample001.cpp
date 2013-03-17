

#include <cil/opencl/opencl.h>
#include <cil/dl/dl.h>
#include <Eigen\Dense>
#include <iostream>
using namespace cil;
using namespace cil::cl;
#include <ctime>

int main()
{

	//int M = 128, N = 1024, P = 1024;  
	 int M =1024*2, N = 1024*6, P = 1024*2;
	//      N               P      
	//|-----------|   |-----------|
	//|           |   |           |
	//|M          | * |N          |
	//|           |   |           |
	//|-----------|   |-----------|

	clock_t begin, end;double elapsed_secs ;
	CLManager& manager = CLManager::getInstance();
	std::cout << "Creating matrices on GPU....";
	begin = clock();
	CLMatrix* M1 = manager.createMatrix(M,N);
	CLMatrix* M2 = manager.createMatrix(N,P);
	CLMatrix* M3 = manager.createMatrix(M,P);
	clFinish(manager.m_gpu_queue);
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;

	std::cout << "Creating matrices on CPU....";
	begin = clock();
	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m1(M,N);
	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m2(N,M);	
	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m3(M,P);	
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;
			
	std::cout << "Filling GPU with random numbers....";
	begin = clock();
	manager.matrixRandomFill(M1);
	manager.matrixRandomFill(M2);
	clFinish(manager.m_gpu_queue);
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;
	
	std::cout << "M3 = M1 * M2... on GPU (Loading Kernels)";
	begin = clock();
	manager.matrix_matrix_multiplication(M1,M2,M3);
	clFinish(manager.m_gpu_queue);
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;

	std::cout << "M3 = M1 * M2... on GPU (3 times)";
	begin = clock();
	for(int k=0;k<3;k++)
		manager.matrix_matrix_multiplication(M1,M2,M3);
	clFinish(manager.m_gpu_queue);
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;
	
	std::cout << "Loading M1, M2 on GPU";
	begin = clock();
	manager.load_gpu_data(M1,m1.data());
	manager.load_gpu_data(M2,m2.data());
	manager.load_gpu_data(M3,m3.data());
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;

	std::cout << "M4 = M1 * M2 on CPU";
	begin = clock();
	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m4 = m1 * m2;
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms] Error:"<< (m3 - m4).array().pow(2).sum() << std::endl;

	//std::cout << m1.topLeftCorner(5,5) << std::endl<< std::endl;
	//std::cout << m2.topLeftCorner(5,5) << std::endl<< std::endl;
	//std::cout << m3.topLeftCorner(5,5) << std::endl<< std::endl;
	//std::cout << m4.topLeftCorner(5,5) << std::endl<< std::endl;
}