#ifndef __CIL_CORE_TYPES_H__
#define __CIL_CORE_TYPES_H__


#include <string>
//#define EIGEN_MATRIXBASE_PLUGIN "cil/core/MatrixBaseAddons.h"


#if (defined WIN32 || defined _WIN32 || defined WINCE) && defined _CILAPI
#  define CIL_EXPORTS __declspec(dllexport)
#else
#  define CIL_EXPORTS
#endif



typedef unsigned char uint8;
typedef unsigned int uint;

namespace cil
{
	
	// The following defines specialized templates to provide a string
	// containing the typename
	template<class T>
	struct TypeName {
	  std::string getName();
	private:
	  T *t; 
	};
 


}

#endif