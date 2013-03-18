
#define MINIBATCH_SIZE_START 16
#define MINIBATCH_SIZE_STEP 256
#define MINIBATCH_SIZE 3601

#define ATTRIBUTE_LENGTH_START 16
#define ATTRIBUTE_LENGTH_STEP 256
#define ATTRIBUTE_LENGTH 3601

#define NETWORK_L1_SIZE 1024
#define N_MULT_EXECUTIONS 10


#include <cil/opencl/opencl.h>
#include <cil/dl/dl.h>
#include <Eigen\Dense>
#include <iostream>
#include <fstream>
using namespace cil;
using namespace cil::cl;
#include <ctime>


int main()
{

	clock_t begin, end;double elapsed_secs ;
	int n =-1, N=0;
	for(int M = MINIBATCH_SIZE_START; M < MINIBATCH_SIZE; M +=MINIBATCH_SIZE_STEP)
		for(int A = ATTRIBUTE_LENGTH_START; A < ATTRIBUTE_LENGTH; A +=ATTRIBUTE_LENGTH_STEP)
			N++;

	int * number_of_elements = new int[N];
	double * create_matrices_gpu = new double[N];
	double * create_matrices_cpu = new double[N];
	double * gpu_random_numbers = new double[N];
	double * gpu_loading_kernel = new double[N];
	double * gpu_mult = new double[N];
	double * load_gpu = new double[N];
	double * cpu_mult = new double[N];
	double * error_s = new double[N];
	//int M =1024*2, N = 1024*6, P = 1024*2;
	//      N               P      
	//|-----------|   |-----------|
	//|           |   |           |
	//|M          | * |N          |
	//|           |   |           |
	//|-----------|   |-----------|

	std::cout << "Cores: "<< Eigen::nbThreads()<< std::endl;
	std::cout << "M: "<< MINIBATCH_SIZE << " N:" << ATTRIBUTE_LENGTH << " P:" << NETWORK_L1_SIZE << std::endl;

	
		
	CLManager& manager = CLManager::getInstance();

	for(int M = MINIBATCH_SIZE_START; M < MINIBATCH_SIZE; M +=MINIBATCH_SIZE_STEP)
	for(int A = ATTRIBUTE_LENGTH_START; A < ATTRIBUTE_LENGTH; A +=ATTRIBUTE_LENGTH_STEP)
	{
		
		std::cout << std::endl << std::endl << "Test #"<< ++n ;
		number_of_elements[n] = M*ATTRIBUTE_LENGTH+A*NETWORK_L1_SIZE;
		std::cout <<" Attributes:"<< A << " Minibatch Size:"<< M << " Elements:"<< number_of_elements[n]<< std::endl;


	std::cout << "Creating matrices on GPU....";
	begin = clock();
	CLMatrix* M1 = manager.createMatrix(M,A);
	CLMatrix* M2 = manager.createMatrix(A,NETWORK_L1_SIZE);
	CLMatrix* M3 = manager.createMatrix(M,NETWORK_L1_SIZE);
	clFinish(manager.m_gpu_queue);
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;
	create_matrices_gpu[n] = elapsed_secs;


	std::cout << "Creating matrices on CPU....";
	begin = clock();
	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m1(M,A);
	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m2(A,NETWORK_L1_SIZE);	
	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m3(M,NETWORK_L1_SIZE);	
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;
	create_matrices_cpu[n] = elapsed_secs;		


	std::cout << "Filling GPU with random numbers....";
	begin = clock();
	manager.matrixRandomFill(M1);
	manager.matrixRandomFill(M2);
	clFinish(manager.m_gpu_queue);
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;
	gpu_random_numbers[n] = elapsed_secs;


	std::cout << "M3 = M1 * M2... on GPU (Loading Kernels)";
	begin = clock();
	manager.matrix_matrix_multiplication(M1,M2,M3);
	clFinish(manager.m_gpu_queue);
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;
	gpu_loading_kernel[n] = elapsed_secs;


	std::cout << "M3 = M1 * M2... on GPU ("<< N_MULT_EXECUTIONS << " times)";
	begin = clock();
	for(int k=0;k<N_MULT_EXECUTIONS;k++)
		manager.matrix_matrix_multiplication(M1,M2,M3);
	clFinish(manager.m_gpu_queue);
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;
	gpu_mult[n] = elapsed_secs / N_MULT_EXECUTIONS;


	std::cout << "Loading M1, M2 on GPU";
	begin = clock();
	manager.load_gpu_data(M1,m1.data());
	manager.load_gpu_data(M2,m2.data());
	manager.load_gpu_data(M3,m3.data());
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]"<< std::endl;
	load_gpu[n] = elapsed_secs;

	std::cout << "M4 = M1 * M2 on CPU";
	begin = clock();
	for(int k=0;k<N_MULT_EXECUTIONS;k++)
		Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m4 = m1 * m2;
	end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "... Done ["<< elapsed_secs*1000 <<"ms]" << std::endl;
	cpu_mult[n] = elapsed_secs/ N_MULT_EXECUTIONS;

	std::cout << "Finding error of M3 and M4";
	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m4 = m1 * m2;

	error_s[n] = (m3 - m4).array().pow(2).sum();
		std::cout << "... Done Error:"<< error_s[n]  << std::endl;
	manager.destroyMatrix(M1);
	manager.destroyMatrix(M2);
	manager.destroyMatrix(M3);
	}

	n = -1;
	std::ofstream MyFile;
    MyFile.open ("output1.csv", std::ios::out) ;
	MyFile << "#Minibatch Size, Attribute Size, Number of elements, create matrices gpu [ms], create matrices cpu[ms], fill_random gpu[ms], load kernel[ms], gpu mult[ms], load data[ms], cpu mult[ms]" << std::endl;

	for(int M = MINIBATCH_SIZE_START; M < MINIBATCH_SIZE; M +=MINIBATCH_SIZE_STEP)
		for(int A = ATTRIBUTE_LENGTH_START; A < ATTRIBUTE_LENGTH; A +=ATTRIBUTE_LENGTH_STEP)
	{
		n++;
		MyFile << M << ", "<< A << ", "<< number_of_elements[n] << ", "
			<< create_matrices_gpu[n] << ", "<< create_matrices_cpu[n] << ", "
			<< gpu_random_numbers[n] << ", "<< gpu_loading_kernel[n]<< ", "
			<< gpu_mult[n] << ", " << load_gpu[n] << ", "<<cpu_mult[n] << "," << std::endl;
	}

	MyFile.close();


	delete[] number_of_elements;
	delete[] create_matrices_gpu;
	delete[] create_matrices_cpu ;
	delete[]gpu_random_numbers ;
delete[] gpu_loading_kernel ;
delete[] gpu_mult ;
	delete[] load_gpu ;
	delete[] cpu_mult ;
	//std::cout << m1.topLeftCorner(5,5) << std::endl<< std::endl;
	//std::cout << m2.topLeftCorner(5,5) << std::endl<< std::endl;
	//std::cout << m3.topLeftCorner(5,5) << std::endl<< std::endl;
	//std::cout << m4.topLeftCorner(5,5) << std::endl<< std::endl;
}