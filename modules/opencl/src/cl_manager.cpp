
#include "cil\opencl\opencl.h"
#include "CL\cl.h"


using namespace cil;
using namespace cl;

CLManager::CLManager()
{

}
CLManager::~CLManager()
{
	cleanup();
}
void CLManager::cleanup()
{
	free(m_platforms_list);
	m_platforms_list=0;
}
bool CLManager::initialize( const char* platform_filter)
{
	bool ok = false;
	int err;                            // error code returned from api calls


    char pPlatformName[128] = { 0 };


	FUNCNAME( "CLManager::initialize" );

	__BEGIN__


	err = clGetPlatformIDs(0, m_platforms_list, &m_num_platforms);            
	if(err != CL_SUCCESS || m_num_platforms==0 )
		ERROR(err,"Couldn't find any platforms");
	
	m_platforms_list = (cl_platform_id *) malloc(sizeof(cl_platform_id) * m_num_platforms);
    if ((err = clGetPlatformIDs(m_num_platforms, m_platforms_list, NULL)) != CL_SUCCESS) {
		ERROR(err, CLErrString(err));
    }
	
	if ( platform_filter )
	for (cl_uint ui = 0; ui < m_num_platforms; ++ui)
    {
        err = clGetPlatformInfo(m_platforms_list[ui], CL_PLATFORM_NAME, 128 * sizeof(char), pPlatformName, NULL);
        if (err == CL_SUCCESS && !strcmp(pPlatformName, platform_filter)) //"Intel(R) OpenCL"
		{
			cl_platform_id temp = m_platforms_list[0];
			m_platforms_list[0] = m_platforms_list[ui];
			m_platforms_list[ui] = temp;

		}

    }
	




	ok = true;
	__END__


	return ok;
}

