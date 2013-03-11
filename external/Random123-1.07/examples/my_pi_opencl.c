/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// Simple OpenCL device kernel and host main program to
// compute pi via random darts at a square

// functions to do boilerplate OpenCL begin and end
#include "util_opencl.h"
#include "pi_check.h"

// Include preprocessed kernel declaration for the array src 
// The GNUmakefile will create pi_opencl_kernel.i in the build
// directory, and then compile this with -I., so use #include <angle> .


const char *progname;
int verbose = 0;
int debug = 0;

int main(int argc, char **argv)
{
     const char *kernelname = "counthits";
    unsigned count =10000;

    cl_int              err;
    cl_context         cl_context;
    cl_program         program;
    cl_kernel          cl_kernel;
    cl_mem          cl_out;
	cl_command_queue    cl_queue;

    size_t i, nthreads, hits_sz;
    size_t cores, work_group_size;
    cl_uint2 *          hits_host;

    double              d = 0.; // timer

    d = timer(&d);
    progname = argv[0];

 
    CHECK(cl::Platform::get(&platformList));        
    CHECKERR(  cl_context = createCLContext(CL_DEVICE_TYPE_GPU,cl_vendor::VENDOR_AMD, &err) );

    std::vector<cl::Device> devices;
    CHECKERR( devices = cl_context.getInfo<CL_CONTEXT_DEVICES>(&err) );


    size_t length = 0;
    const char * sourceStr = loadFileToString("pi_opencl_kernel.ocl","",&length);

    cl::Program::Sources sources(1, std::make_pair(sourceStr, length));
    program = cl::Program(cl_context, sources);

    CHECK( program.build(devices,"-I ..\\include") );

    CHECKERR(work_group_size = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err) );
    CHECKERR(cores = devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err) );
    cores *= 16*4; //Tahiti.

    if (work_group_size > 64) work_group_size /= 2;
    nthreads = cores * work_group_size*32; //2048*128 = 262144

    if (count == 0)
    count = NTRIES/nthreads; //38

    printf("Count: %lu\n",count);



    hits_sz = nthreads * sizeof(hits_host[0]);//2097152
    CHECKNOTZERO(hits_host = (cl_uint2 *)malloc(hits_sz));

    CHECKERR    ( cl_out = cl::Buffer(  cl_context,  CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, hits_sz, hits_host, &err));
    CHECKERR    ( cl_kernel = cl::Kernel(program,kernelname,&err) );
    CHECK       ( cl_kernel.setArg( 0, count) );
    CHECK       ( cl_kernel.setArg( 1, cl_out) );

    CHECKERR (cl_queue = cl::CommandQueue(cl_context, devices[0], 0, &err) );
    cl::Event event;

    CHECK( cl_queue.enqueueNDRangeKernel(cl_kernel,cl::NullRange,cl::NDRange(nthreads), cl::NDRange(work_group_size), NULL,  &event) );
    event.wait();
    CHECK( cl_queue.enqueueReadBuffer(cl_out, CL_TRUE, 0,hits_sz, hits_host) );

    unsigned long hits = 0, tries = 0;
    for (i = 0; i < nthreads; i++) {
#ifdef _DEBUG   
        printf("%lu %u %u\n", (unsigned long)i, hits_host[i].s[0], hits_host[i].s[1]);
#endif
    hits += hits_host[i].s[0];
    tries += hits_host[i].s[1];
    }


    return pi_check(hits, tries);
}
