#ifndef __CIL_ALGORITHMS_ALGORITHMS_H__
#define __CIL_ALGORITHMS_ALGORITHMS_H__

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cil/core/types_c.h>
#include <iostream>

namespace cil
{
	namespace alg
	{
		CIL_EXPORTS void test();
		
		
		//template <typename Derived>
		//int gpa(const Eigen::MatrixBase<Derived>& data)
		//{
		//	return 1;
		//	int number_of_landmarks = data.cols()/2;
		//	int number_of_shapes	= data.rows();
		//
		//

		//	//Complex notation and Substracting Mean.
		//	
		//	//typename Derived::
		//	
		////   Eigen::Matrix<typename Derived::PlainObject,-1,-1> X(number_of_shapes, number_of_landmarks);
		////	X.real() = data.leftCols(number_of_landmarks);
		//	//X.imag() = data.rightCols(number_of_landmarks);

		//	//int n_samples = Data.rows();
		//	//int n_attributes = Data.cols();
		////	int n = sizeof(Derived);
		////	gpa(Data);
		//	//Eigen::MatrixXcf
		//	return 1;
		//};
		//template<>
		//int gpa(const Eigen::MatrixBase<Eigen::MatrixXf> &m)
		//{
		//	return 3;
		//}
		//template<>
		//int gpa(const Eigen::MatrixBase<Eigen::MatrixXd> & data)
		//{
		//	return 4;
		//}


		/*
		*	This function performs a (full) generalized Procrustes analysis in 2D.
		*	X should be a M by N matrix with M observations of N Landmarks in Complex Notation.
		*/
		template <typename Derived>
		void gpa(const Eigen::MatrixBase<Derived>& X, Eigen::MatrixBase<Derived> const & C_, Eigen::MatrixBase<Derived> const & mean_ )
		{
			typedef Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> MatCplx;
			//typedef Matrix<std::complex<typename A::RealScalar, Dynamic, Dynamic> MatCplx;

			Eigen::MatrixBase<Derived>& C = const_cast< Eigen::MatrixBase<Derived>& >(C_);
			C.derived().resize(X.rows(),X.cols()); // resize the derived object
			//assert(C.rows() == X.Cols());

			Eigen::MatrixBase<Derived>& m = const_cast< Eigen::MatrixBase<Derived>& >(mean_);
			m.derived().resize(X.cols(),1); // resize the derived object

			int n;

			//Solve EigenProblem
			Eigen::ComplexEigenSolver<Derived> solver;
			solver.compute((X.transpose()*X.conjugate()));
			
			// Take the last EigenVector, its sorted for maximum variance and its normalized already.
			m = solver.eigenvectors().col(solver.eigenvectors().cols()-1);
			//m = m / m.norm();
			
			
			// Full Procrusters fits
			// Calculate fits. Find a M by 1, M = landmarks, matrix f.
			MatCplx f = ( X* m.conjugate()).cwiseQuotient(X.conjugate().cwiseProduct(X).rowwise().sum());

			//Mean transformation scaled to unit size -> rotation only.
			typename Derived::Scalar mf = f.mean();
			mf = mf / abs(mf); //sqrt(mf.real()*mf.real()+mf.imag()*mf.imag());
	
			//Transform each observation.
			//f= f.conjugate();
			//for(n=0;n<X.cols();++n)
 			//	C.col(n) = X.col(n).cwiseProduct(f);
			C = f.conjugate().asDiagonal() * X;

			//Rotate all observations and mean based on mean transforms.
			C = C*mf;
			m = m * mf;

		}



	}
}

//			MatCplx f = (m.adjoint() * C).array() / (C.adjoint() * C).diagonal().adjoint().array();
		//	MatCplx f = (X * m).cwiseQuotient((X * X.adjoint()).diagonal());
			//MatCplx f = (m.adjoint() * C).cwiseQuotient(C.adjoint().cwiseProduct(C.transpose()).rowwise().sum().adjoint());
			
			//MatCplx f = (X * eigvec_mean).array() / ((X.array().rowwise() * X.array().rowwise()).sum());
			//Transform
#endif