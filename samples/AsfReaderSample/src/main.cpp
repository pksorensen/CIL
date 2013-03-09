

#include <iostream>
#include <complex>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cil/algorithms/algorithms.h>

	struct AsfRow
	{
		int path_nr;
		int type;
		float x_rel;
		float y_rel;
		int point_nr;
		int connects_from;	
		int connects_to;
	};
	typedef Eigen::Matrix<float,7,-1, Eigen::ColMajor> AsfMatrix;
		int readAsf2Eigen(char* path, AsfMatrix & data)
	{
		FILE *fp = fopen(path,"r");
		if(fp==0)
			return -1;

		int datapoints;
		float *dst=0;
		char buf[255];

		//Find numbers of points.
		while( (fgets(buf,255,fp) != 0 && buf[0] == '#') || sscanf(buf,"%d",&datapoints)==-1);

		
		data.resize(7,datapoints); //Eigen::Matrix<float,7,-1> & data
		dst = data.data();

		while( fgets(buf,255,fp) != 0 )
		{
			
			int n = sscanf(buf,"%f \t%f \t%f \t%f \t%f \t%f \t%f",dst, dst+1,dst+2,dst+3,dst+4,dst+5,dst+6);
			dst += (n<=0)?0:7;
			
		}

		fclose(fp);
	
		return !(datapoints == (dst - data.data()-7)/7);


	}

int main()
{
	float width = 640.0f, height=480.0f;
	char buf[255];
	int n, pose; 
	AsfMatrix data;

	Eigen::Matrix<float,40*6,-1> Shapes; //240x116

	float mean_size = 0;
	for(n=0;n<40;n++)
	{
		for(pose=0;pose<6;pose++)
		{
			sprintf(buf,"../../../CIL/data/imm_face_db/%.2d-%dm.asf",n+1,pose+1);
			if(readAsf2Eigen(buf, data) != 0)
			{
				sprintf(buf,"../../../CIL/data/imm_face_db/%.2d-%df.asf",n+1,pose+1);
				if (readAsf2Eigen(buf, data) != 0)
					continue;		
			}

			//Initialize The Shapes Container
			if(Shapes.cols() == 0)
				Shapes.resize(40*6,data.cols()*2);
			
			

			//Copy the found data
			Shapes.block(n*6+pose,0,1,data.cols()) = data.row(2) * width;
			Shapes.block(n*6+pose,data.cols(),1,data.cols()) = data.row(3) * height;

			

			//Compute MeanShape
			auto mean_x		= Shapes.block(n*6+pose,0,1,data.cols()).mean();
			auto mean_y		= Shapes.block(n*6+pose,data.cols(),1,data.cols()).mean();
			auto mshape_x	= (Shapes.block(n*6+pose,0,1,data.cols()).array()-mean_x).pow(2) ;
			auto mshape_y	= (Shapes.block(n*6+pose,data.cols(),1,data.cols()).array()-mean_y).pow(2) ;
			mean_size		+= sqrt((mshape_x+mshape_y).sum());

		
			//std::cout << Shapes.block(n*pose+pose,0,1,data.cols()) << std::endl;
		//		std::cout << Shapes.block(0,0,1,5) << std::endl ;
	//std::cout << Shapes.block(1,0,1,5) << std::endl ;
	//std::cout << Shapes.block(2,0,1,5) << std::endl ;
	//std::cout << Shapes.block(3,0,1,5) << std::endl << std::endl;
		}

	}
	mean_size /= 40*6;
	int number_of_landmarks = data.cols();
	int number_of_shapes	= Shapes.rows();

	//Complex notation and Substracting Mean.
	Eigen::MatrixXcf X(number_of_shapes, number_of_landmarks);
	X.real() = Shapes.leftCols(number_of_landmarks);
	X.imag() = Shapes.rightCols(number_of_landmarks);
	X.array().colwise() -= X.rowwise().mean().array();

	//Eigen::MatrixXcd XX(10,10);

	//double test[10] = {0};
	//Eigen::Map<Eigen::MatrixXd> mat(test, 10, 1);
	Eigen::MatrixXcf C;
	Eigen::MatrixXcf Mean;
	cil::alg::gpa(X,C,Mean);
	std::cout << X.rows() << " , " << X.cols() << std::endl<< std::endl;
	std::cout << C.rows() << " , " << C.cols() << std::endl<< std::endl;
	std::cout << Mean.rows() << " , " << Mean.cols() << std::endl<< std::endl;
	std::cout << C.row(1).transpose() << std::endl<< std::endl;
	return 0;

	X.array().colwise() -= X.rowwise().mean().array();

	//Eigen::MatrixXcf X = Shapes.block(0,0,Shapes.rows(),data.cols()) * std::complex<float>(0,1) +
	//	Shapes.block(0,data.cols(),Shapes.rows(),data.cols())*std::complex<float>(1,0);
	//Eigen::VectorXcf Mean = X.rowwise().mean();

	//std::complex<float> *m_ptr = Mean.data();
	//for(n=0;n<Mean.rows();++n)
	//	X.row(n) = X.row(n).array() - *m_ptr++;

	//Solve Eigen Problem
	Eigen::MatrixXcf A = X.transpose().conjugate() * X;
	Eigen::ComplexEigenSolver<Eigen::MatrixXcf> solver;
	solver.compute(A);
	
//	std::cout << "The Eigenvales of A are:" << std::endl << solver.eigenvalues() <<std::endl<<std::endl;
//	std::complex<float> lambda = solver.eigenvalues()[57];
//	std::cout << "Consider the first eigenvalue, lambda = " << lambda << std::endl;
//	std::cout << "EigenVec for largest EigenVals of A are:" << std::endl << solver.eigenvectors().col(57) <<std::endl<<std::endl;

	auto eigvec_mean = solver.eigenvectors().col(solver.eigenvectors().cols()-1);

	// Full Procrusters fits
	Eigen::MatrixXcf f = (X * eigvec_mean).array() / (X * X.transpose().conjugate()).diagonal().array();

	//Transform
	
	auto f_conj = f.conjugate().array();
	for(n=0;n<X.cols();++n)
		X.col(n) = X.col(n).array() * f_conj;
	auto mf = f.mean();
	std::cout << mf << std::endl<< std::endl;
	mf = mf / sqrt(mf.real()*mf.real()+mf.imag()*mf.imag());
	std::cout << mf << std::endl<< std::endl;
	auto m = eigvec_mean * mf;
	X = X*mf;



	std::cout << X.row(0).transpose() << std::endl<< std::endl;

	return 0;
}