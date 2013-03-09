#ifndef __CIL_CORE_H
#define __CIL_CORE_H

#include "cil/core/types_c.h"
#include <Eigen/Eigen>




namespace cil
{
	#define __BEGIN__       {
	#define __END__         goto exit; exit: ; }
	#define __EXIT__        goto exit



	extern "C" CIL_EXPORTS void __cdecl error( int status, const char* func_name,
                    const char* err_msg, const char* file_name, int line );
	#define FUNCNAME( Name )  \
		static char cvFuncName[] = Name

	#define ERROR( Code, Msg )                                      \
	{                                                               \
		cil::error( (Code), cvFuncName, Msg, __FILE__, __LINE__ );       \
		__EXIT__;                                                   \
	}




	typedef Eigen::Matrix<size_t,-1,1> IndexVector;

	template <typename Derived>
	void indexes(Eigen::MatrixBase<Derived>& X)
	{
		typedef typename Derived::Scalar Type;
		Type start, end = X.rows();
		Type* ptr = X.derived().data();
		for(start = 0;start < end;*ptr++=start++);
	}

}

#endif