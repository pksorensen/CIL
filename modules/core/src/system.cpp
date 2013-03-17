
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


char* cil::file2string(const char* filename, const char* preamble, size_t* final_length) {
        FILE * file_stream = NULL;
        size_t source_length;

        // open the OpenCL source code file
  file_stream = fopen(filename, "rb");
  if(!file_stream) return NULL;
        //if(fopen_s(&file_stream, filename, "rb") != 0) return NULL;

        size_t preamble_length = strlen(preamble);

        // get the length of the source code
        fseek(file_stream, 0, SEEK_END); 
        source_length = ftell(file_stream);
        fseek(file_stream, 0, SEEK_SET); 

        // allocate a buffer for the source code string and read it in
        char* source_str = (char *)malloc(source_length + preamble_length + 1); 
        memcpy(source_str, preamble, preamble_length);
        if (fread((source_str) + preamble_length, source_length, 1, file_stream) != 1) {
                fclose(file_stream);
                free(source_str);
                return 0;
        }

        // close the file and return the total length of the combined (preamble + source) string
        fclose(file_stream);
        if(final_length != 0) 
                *final_length = source_length + preamble_length;
        source_str[source_length + preamble_length] = '\0';

        return source_str;
};