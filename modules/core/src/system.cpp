
#include "cil/core/core.h"
#include <iostream>

extern "C" void cil::error( int code, const char* func_name,
                      const char* err_msg,
                      const char* file_name, int line )
{
	fprintf(stderr,"%d > Func: %s, Error:%s, File:%s at line %d",
		code,func_name, err_msg,file_name,line);
}