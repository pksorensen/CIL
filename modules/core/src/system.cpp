
#include "cil/core/core.h"
#include <iostream>
using namespace cil;
extern "C" void cil::error( int code, const char* func_name,
                      const char* err_msg,
                      const char* file_name, int line )
{
	fprintf(stderr,"%d > Func: %s, Error:%s, File:%s at line %d",
		code,func_name, err_msg,file_name,line);
}

template<> std::string TypeName<double>::getName() {return(std::string("double")); }
template<> std::string TypeName<float>::getName() {return(std::string("float")); }
template<> std::string TypeName<unsigned long>::getName() {return(std::string("ulong"));}
template<> std::string TypeName<long>::getName() { return(std::string("long")); }
template<> std::string TypeName<unsigned int>::getName() {return(std::string("uint"));}
template<> std::string TypeName<int>::getName() {return(std::string("int")); }
template<> std::string TypeName<unsigned char>::getName() {return(std::string("uchar"));}
template<> std::string TypeName<char>::getName() {return(std::string("char")); }