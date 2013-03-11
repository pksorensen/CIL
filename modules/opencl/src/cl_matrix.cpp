
#include <cil/opencl/opencl.h>

using namespace cil::cl;

CLMatrix::CLMatrix(CLManager * manager, cl_uint m, cl_uint n)
	: m_manager(manager), m_rows(m), m_cols(n), status(-1)
{
	
}
CLMatrix::~CLMatrix()
{
	
}