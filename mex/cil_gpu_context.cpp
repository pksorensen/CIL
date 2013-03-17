#include <mex.h>
#include <stdint.h>
#include <string>
#include <cil/opencl/opencl.h>

#ifndef VERBOSE_DEBUG
#define VERBOSE_DEBUG 0
#endif
#define VERBOSE_PRINT(X,ERR) \
	if ((ERR = X) && VERBOSE_DEBUG) \
		mexPrintf(cil::cl::CLErrString(ERR)); 
		
		
		

#define CLASS_HANDLE_SIGNATURE 0xFF00F0A5
template<class base> class class_handle
{
public:
class_handle(base *ptr) : ptr_m(ptr), name_m(typeid(base).raw_name()) { signature_m = CLASS_HANDLE_SIGNATURE; };
~class_handle() { signature_m = 0; delete ptr_m; };
bool isValid() { return ((signature_m == CLASS_HANDLE_SIGNATURE) && !strcmp(name_m.c_str(), typeid(base).raw_name())); };
base *ptr() { return ptr_m; };

private:
uint32_t signature_m;
std::string name_m;
base *ptr_m;
};

template<class base> inline mxArray *convertPtr2Mat(base *ptr)
{
	
//mexLock();
mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
*((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(new class_handle<base>(ptr));
return out;
};

template<class base>  class_handle<base> *convertMat2HandlePtr(const mxArray *in)
{
if (mxGetNumberOfElements(in) != 1 || mxGetClassID(in) != mxUINT64_CLASS || mxIsComplex(in))
mexErrMsgTxt("Input must be a real uint64 scalar.");
class_handle<base> *ptr = reinterpret_cast<class_handle<base> *>(*((uint64_t *)mxGetData(in)));
if (!ptr->isValid())
mexErrMsgTxt("Handle not valid.");
return ptr;
};

template<class base> inline base *convertMat2Ptr(const mxArray *in)
{
return convertMat2HandlePtr<base>(in)->ptr();
};

template<class base> inline void destroyObject(const mxArray *in)
{
//delete convertMat2HandlePtr<base>(in);
//mexUnlock();
};

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
	char *buf;
	int   buflen;
	int status;
	
   
    #define HANDLE prhs[0]    
    #define METHOD prhs[1] 
	if( nrhs == 0)
	{
		
		//cil::cl::CLManager& m=;
		
		mexPrintf("HEK");
		plhs[0] = convertPtr2Mat<cil::cl::CLManager>(&cil::cl::CLManager::getInstance());
		return;
	}else if (nrhs ==1)
	{
		
		plhs[0] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
		*((int32_t *)mxGetData(plhs[0])) =  cil::cl::CLManager::count; //cil::cl::CLManager::getInstance().m_seed;
		
	}
	
	if (nrhs >= 2)
	{
		if(!mxIsChar(METHOD))
			 mexErrMsgTxt("Second argument should be a method");
		cil::cl::CLManager* cl=convertMat2Ptr<cil::cl::CLManager>(HANDLE);

		buflen = mxGetM(METHOD) * mxGetN(METHOD) + 1;
		buf = (char*)mxCalloc(buflen, sizeof(char));
		if (buf == NULL)
			mexErrMsgTxt("Not enough heap space to hold converted string.");
		status = mxGetString(METHOD, buf, buflen);

		if (status == 0)
			mexPrintf("The converted string is \n%s.\n", buf);
		else
			mexErrMsgTxt("Could not convert string data.");
	

		if (0==strcmpi(buf,"clear"))
		{
			destroyObject<cil::cl::CLManager>(HANDLE);

		}else if(0==strcmpi(buf,"createGPUMatrix"))
		{
			if (!mxIsSingle(prhs[2]))
				mexErrMsgTxt("Need to be single");
			
			size_t rows = mxGetM(prhs[2]);
			size_t cols = mxGetN(prhs[2]);
			float* data = (float*)mxGetData(prhs[2]);
			
			if(VERBOSE_DEBUG)
				mexPrintf("Creating Matrix<%d,%d>[%f %f %f ...] on GPU\n",
				rows,cols,data[0], data[1], data[2]);

			plhs[0] = convertPtr2Mat<cil::cl::CLMatrix>(
				cl->createMatrix(rows,cols,data));
	
		}else if(0==strcmpi(buf,"GPUMatrixMult"))
		{
			if (nrhs < 3)
				mexErrMsgTxt("Need atleast two inputs");
			int err;

				if(!cl->isloaded_mm_mult_kernel())					
					VERBOSE_PRINT(cl->load_mm_mult_kernel(),err)

				cil::cl::CLMatrix* A=convertMat2Ptr<cil::cl::CLMatrix>( prhs[2] );
				cil::cl::CLMatrix* B=convertMat2Ptr<cil::cl::CLMatrix>( prhs[3] );
				cil::cl::CLMatrix* C = 0;
				if (nrhs>3)
				{
					C=convertMat2Ptr<cil::cl::CLMatrix>( prhs[4] );
				}
				else
				{
					plhs[0] = convertPtr2Mat<cil::cl::CLMatrix>(
						C=cl->createMatrix(A->m_rows,B->m_cols));
				}

				mexPrintf("Stats: %s",cil::cl::CLErrString(
					cl->matrix_matrix_multiplication(A,B,C)));
				
			
			
		}else if(0==strcmpi(buf,"GPUWaitOnCalculations"))
		{
			if(VERBOSE_DEBUG)
				mexPrintf("Stall until all commands are finished..\n");
			clFinish(cl->m_gpu_queue);
		
		}else if(0==strcmpi(buf,"loadGPUMatrix"))
		{
			if (nrhs <2)
				mexErrMsgTxt("Need atleast one inputs");

			cil::cl::CLMatrix* A=convertMat2Ptr<cil::cl::CLMatrix>( prhs[2] );

			plhs[0] = mxCreateNumericMatrix( A->m_rows,A->m_cols, mxSINGLE_CLASS, mxREAL);
			mexPrintf("Stats: %s",cil::cl::CLErrString(cl->load_gpu_data(A,
				(float*)mxGetData(plhs[0]))));
		}
		

	}
		//	 mexErrMsgIdAndTxt( "MATLAB:gpa:inputNotDefined",
    //        "nargin should be 2");


}
