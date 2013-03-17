

#include <cil/opencl/opencl.h>
#include <cil/dl/dl.h>
#include <cil/algorithms/algorithms.h>
#include <Eigen\Dense>
#include <iostream>
using namespace cil;
using namespace cil::cl;
#include <ctime>
#include <fstream>

class MNISTTrainingData : public dl::TrainingData
{
public:
	MNISTTrainingData() :
		TrainingData(60000,28*28,1,100),
		m_train_data(60000,28*28),
		m_train_label(60000,1),
		m_test_data(10000,28*28),
		m_test_label(10000,1)

	{
		int n=0;
		char* folder = "C:/Development/CIL/data/mnist/";
		std::string train_data = std::string(folder) +"train-images-idx3-ubyte";
		std::string train_label = std::string(folder) +"train-labels-idx1-ubyte";
		std::string test_data = std::string(folder) +"t10k-images-idx3-ubyte";
		std::string test_label = std::string(folder) +"t10k-labels-idx1-ubyte";

		//// Represent MNIST datafiles as C++ file streams f1 and f2 respectively
        std::ifstream f1(train_data,std::ios::in | std::ios::binary); // image data
        std::ifstream f2(train_label,std::ios::in | std::ios::binary);		 // label data
		std::ifstream f3(test_data,std::ios::in | std::ios::binary); // image data
        std::ifstream f4(test_label,std::ios::in | std::ios::binary);		 // label data
		
		char *buffer = new char[16];

		f1.read(buffer,16);
		f2.read(buffer,8);		
		f3.read(buffer,16);
		f4.read(buffer,8);
		

		Eigen::Matrix<unsigned char,-1,-1,Eigen::RowMajor> u_train_data(60000,28*28);
		Eigen::Matrix<unsigned char,-1,-1,Eigen::RowMajor> u_train_label(60000,1);
		Eigen::Matrix<unsigned char,-1,-1,Eigen::RowMajor> u_test_data(10000,28*28);
		Eigen::Matrix<unsigned char,-1,-1,Eigen::RowMajor> u_test_label(10000,1);

		f1.read((char*)u_train_data.data(),60000*28*28);			
		f2.read((char*)u_train_label.data(),60000);
		f3.read((char*)u_test_data.data(),10000*28*28);			
		f4.read((char*)u_test_label.data(),10000);
			

		delete[] buffer;
		f1.close();
		f2.close();
		f3.close();
		f4.close();
		
		
		Eigen::Matrix<unsigned char,-1,-1,Eigen::RowMajor> a(1,10);
		a << 0, 1,2,3,4,5,6,7,8,9;

		
		//std::cout << m_train_label	<< std::endl;

		m_train_data = u_train_data.cast<float>() / 255;
		m_train_label = (u_train_label.replicate(1,10).array() == a.replicate(60000,1).array()).cast<float>();
		m_test_data=u_test_data.cast<float>() / 255;
		m_test_label= (u_test_label.replicate(1,10).array() == a.replicate(10000,1).array()).cast<float>();

		//std::cout << m_train_label.topRows(100) << std::endl << std::endl;
		//std::cout << m_train_data.row(5) << std::endl << std::endl;
		
	/*	Eigen::MatrixXf t = data.row(0).cast<float>();
		t.resize(28,28);
		std::cout << t.transpose() << std::endl;

		float v = (float) *(data.data()+28*28+28*5+17);
		float v1 = (float) *(data.row(1).data()+28*5+17);
		std::cout << v << std::endl;
	std::cout << v1 << std::endl;*/
		
	}

	virtual const float* train_data() {return m_train_data.data();};
	virtual const float* train_label_data() {return m_train_label.data();};	
private:

	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m_train_data;
	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m_train_label;
	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m_test_data;
	Eigen::Matrix<float,-1,-1,Eigen::RowMajor> m_test_label;
};


#define PRINT 0
int main()
{
	using namespace std;
	srand (time(NULL));
	
	MNISTTrainingData data;

	
	IndexVector idx(100);
	indexes(idx);

	std::cout << idx << std::endl;
	alg::knuth_shuffle(idx);
	std::cout << idx << std::endl;


	
	CLManager& manager = CLManager::getInstance();

	if (manager.status != 0)
		std::cout << manager.status << std::endl;

	int arc[] = {784, 500, 500, 2000 , 10};
	dl::DLParams params(arc,5);
//	params.output = CIL_DL_LINEAR;


	dl::NeuralNetwork nn(params);
	nn.train(data);

//	cl::CLMatrix* m = nn.get_weights().at(3);
	//Eigen::MatrixXf mat = m->load_from_gpu();
	//std::cout << mat.row(0) << std::endl;
//	std::cout << m->m_rows << " " << m->m_cols<<std::endl;

	cl::CLMatrix* m1 = nn.get_activations().at(3);
	Eigen::MatrixXf act = m1->load_from_gpu();
	std::cout << act << std::endl<<std::endl;
	//std::cout << m1->m_rows << " " << m1->m_cols<<std::endl;



	
	//Eigen::MatrixXf datainput = data.gpu_load_minibatch(0)->load_from_gpu();
	//std::cout << datainput.row(0) << std::endl << std::endl;
	//std::cout << datainput.row(5) << std::endl << std::endl;

	Eigen::MatrixXf errors = nn.get_elementwise_error()->load_from_gpu();
	std::cout << errors << std::endl <<errors.sum()/100/2 << std::endl;

	return 0;
}