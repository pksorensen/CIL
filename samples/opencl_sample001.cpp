//
// File:       hello.c
//
// Abstract:   A simple "Hello World" compute example showing basic usage of OpenCL which
//             calculates the mathematical square (X[i] = pow(X[i],2)) for a buffer of
//             floating point values.
//             
//
// Version:    <1.0>
//
// Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple Inc. ("Apple")
//             in consideration of your agreement to the following terms, and your use,
//             installation, modification or redistribution of this Apple software
//             constitutes acceptance of these terms.  If you do not agree with these
//             terms, please do not use, install, modify or redistribute this Apple
//             software.
//
//             In consideration of your agreement to abide by the following terms, and
//             subject to these terms, Apple grants you a personal, non - exclusive
//             license, under Apple's copyrights in this original Apple software ( the
//             "Apple Software" ), to use, reproduce, modify and redistribute the Apple
//             Software, with or without modifications, in source and / or binary forms;
//             provided that if you redistribute the Apple Software in its entirety and
//             without modifications, you must retain this notice and the following text
//             and disclaimers in all such redistributions of the Apple Software. Neither
//             the name, trademarks, service marks or logos of Apple Inc. may be used to
//             endorse or promote products derived from the Apple Software without specific
//             prior written permission from Apple.  Except as expressly stated in this
//             notice, no other rights or licenses, express or implied, are granted by
//             Apple herein, including but not limited to any patent rights that may be
//             infringed by your derivative works or by other works in which the Apple
//             Software may be incorporated.
//
//             The Apple Software is provided by Apple on an "AS IS" basis.  APPLE MAKES NO
//             WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED
//             WARRANTIES OF NON - INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
//             PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION
//             ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
//
//             IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR
//             CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//             SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//             INTERRUPTION ) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION
//             AND / OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER
//             UNDER THEORY OF CONTRACT, TORT ( INCLUDING NEGLIGENCE ), STRICT LIABILITY OR
//             OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright ( C ) 2008 Apple Inc. All Rights Reserved.
//

////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>
#include <math.h>
//#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cil/opencl/opencl.h>
using namespace cil::cl;

#define Warning(...)    fprintf(stderr, __VA_ARGS__)
////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define DATA_SIZE (1024)

////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the square of an input array 
//
const char *KernelSource = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////



/*
 * PrintDevice --
 *
 *      Dumps everything about the given device ID.
 *
 * Results:
 *      void.
 */

static void
PrintDevice(cl_device_id device) {
#define LONG_PROPS \
  defn(VENDOR_ID), \
  defn(MAX_COMPUTE_UNITS), \
  defn(MAX_WORK_ITEM_DIMENSIONS), \
  defn(MAX_WORK_GROUP_SIZE), \
  defn(PREFERRED_VECTOR_WIDTH_CHAR), \
  defn(PREFERRED_VECTOR_WIDTH_SHORT), \
  defn(PREFERRED_VECTOR_WIDTH_INT), \
  defn(PREFERRED_VECTOR_WIDTH_LONG), \
  defn(PREFERRED_VECTOR_WIDTH_FLOAT), \
  defn(PREFERRED_VECTOR_WIDTH_DOUBLE), \
  defn(MAX_CLOCK_FREQUENCY), \
  defn(ADDRESS_BITS), \
  defn(MAX_MEM_ALLOC_SIZE), \
  defn(IMAGE_SUPPORT), \
  defn(MAX_READ_IMAGE_ARGS), \
  defn(MAX_WRITE_IMAGE_ARGS), \
  defn(IMAGE2D_MAX_WIDTH), \
  defn(IMAGE2D_MAX_HEIGHT), \
  defn(IMAGE3D_MAX_WIDTH), \
  defn(IMAGE3D_MAX_HEIGHT), \
  defn(IMAGE3D_MAX_DEPTH), \
  defn(MAX_SAMPLERS), \
  defn(MAX_PARAMETER_SIZE), \
  defn(MEM_BASE_ADDR_ALIGN), \
  defn(MIN_DATA_TYPE_ALIGN_SIZE), \
  defn(GLOBAL_MEM_CACHELINE_SIZE), \
  defn(GLOBAL_MEM_CACHE_SIZE), \
  defn(GLOBAL_MEM_SIZE), \
  defn(MAX_CONSTANT_BUFFER_SIZE), \
  defn(MAX_CONSTANT_ARGS), \
  defn(LOCAL_MEM_SIZE), \
  defn(ERROR_CORRECTION_SUPPORT), \
  defn(PROFILING_TIMER_RESOLUTION), \
  defn(ENDIAN_LITTLE), \
  defn(AVAILABLE), \
  defn(COMPILER_AVAILABLE),

#define STR_PROPS \
  defn(NAME), \
  defn(VENDOR), \
  defn(PROFILE), \
  defn(VERSION), \
  defn(EXTENSIONS),

#define HEX_PROPS \
   defn(SINGLE_FP_CONFIG), \
   defn(QUEUE_PROPERTIES),


/* XXX For completeness, it'd be nice to dump this one, too. */
#define WEIRD_PROPS \
   CL_DEVICE_MAX_WORK_ITEM_SIZES,

   static struct { cl_device_info param; const char *name; } longProps[] = {
#define defn(X) { CL_DEVICE_##X, #X }
      LONG_PROPS
#undef defn
      { 0, NULL },
   };

   static struct { cl_device_info param; const char *name; } hexProps[] = {
#define defn(X) { CL_DEVICE_##X, #X }
      HEX_PROPS
#undef defn
      { 0, NULL },
   };
   static struct { cl_device_info param; const char *name; } strProps[] = {
#define defn(X) { CL_DEVICE_##X, #X }
      STR_PROPS
#undef defn
      { CL_DRIVER_VERSION, "DRIVER_VERSION" },
      { 0, NULL },
   };
   cl_int status;
   size_t size;
   char buf[65536];
   long long val; /* Avoids unpleasant surprises for some params */
   int ii;


   for (ii = 0; strProps[ii].name != NULL; ii++) {
      status = clGetDeviceInfo(device, strProps[ii].param, sizeof buf, buf, &size);
      if (status != CL_SUCCESS) {
         Warning("\tdevice[%p]: Unable to get %s: %s!\n",
                 device, strProps[ii].name, CLErrString(status));
         continue;
      }
      if (size > sizeof buf) {
         Warning("\tdevice[%p]: Large %s (%d bytes)!  Truncating to %d!\n",
                 device, strProps[ii].name, size, sizeof buf);
      }
      printf("\tdevice[%p]: %s: %s\n",
             device, strProps[ii].name, buf);
   }
   printf("\n");

   status = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof val, &val, NULL);
   if (status == CL_SUCCESS) {
      printf("\tdevice[%p]: Type: ", device);
      if (val & CL_DEVICE_TYPE_DEFAULT) {
         val &= ~CL_DEVICE_TYPE_DEFAULT;
         printf("Default ");
      }
      if (val & CL_DEVICE_TYPE_CPU) {
         val &= ~CL_DEVICE_TYPE_CPU;
         printf("CPU ");
      }
      if (val & CL_DEVICE_TYPE_GPU) {
         val &= ~CL_DEVICE_TYPE_GPU;
         printf("GPU ");
      }
      if (val & CL_DEVICE_TYPE_ACCELERATOR) {
         val &= ~CL_DEVICE_TYPE_ACCELERATOR;
         printf("Accelerator ");
      }
      if (val != 0) {
         printf("Unknown (0x%llx) ", val);
      }
      printf("\n");
   } else {
      Warning("\tdevice[%p]: Unable to get TYPE: %s!\n",
              device, CLErrString(status));
   }

   status = clGetDeviceInfo(device, CL_DEVICE_EXECUTION_CAPABILITIES,
                            sizeof val, &val, NULL);
   if (status == CL_SUCCESS) {
      printf("\tdevice[%p]: EXECUTION_CAPABILITIES: ", device);
      if (val & CL_EXEC_KERNEL) {
         val &= ~CL_EXEC_KERNEL;
         printf("Kernel ");
      }
      if (val & CL_EXEC_NATIVE_KERNEL) {
         val &= ~CL_EXEC_NATIVE_KERNEL;
         printf("Native ");
      }
      if (val) {
         printf("Unknown (0x%llx) ", val);
      }
      printf("\n");
   } else {
      Warning("\tdevice[%p]: Unable to get EXECUTION_CAPABILITIES: %s!\n",
              device, CLErrString(status));
   }

   status = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                            sizeof val, &val, NULL);
   if (status == CL_SUCCESS) {
      static const char *cacheTypes[] = { "None", "Read-Only", "Read-Write" };
      static int numTypes = sizeof cacheTypes / sizeof cacheTypes[0];

      printf("\tdevice[%p]: GLOBAL_MEM_CACHE_TYPE: %s (%lld)\n",
             device, val < numTypes ? cacheTypes[val] : "???", val);
   } else {
      Warning("\tdevice[%p]: Unable to get GLOBAL_MEM_CACHE_TYPE: %s!\n",
              device, CLErrString(status));
   }
   status = clGetDeviceInfo(device,
                            CL_DEVICE_LOCAL_MEM_TYPE, sizeof val, &val, NULL);
   if (status == CL_SUCCESS) {
      static const char *lmemTypes[] = { "???", "Local", "Global" };
      static int numTypes = sizeof lmemTypes / sizeof lmemTypes[0];

      printf("\tdevice[%p]: CL_DEVICE_LOCAL_MEM_TYPE: %s (%lld)\n",
             device, val < numTypes ? lmemTypes[val] : "???", val);
   } else {
      Warning("\tdevice[%p]: Unable to get CL_DEVICE_LOCAL_MEM_TYPE: %s!\n",
              device, CLErrString(status));
   }

   for (ii = 0; hexProps[ii].name != NULL; ii++) {
      status = clGetDeviceInfo(device, hexProps[ii].param, sizeof val, &val, &size);
      if (status != CL_SUCCESS) {
         Warning("\tdevice[%p]: Unable to get %s: %s!\n",
                 device, hexProps[ii].name, CLErrString(status));
         continue;
      }
      if (size > sizeof val) {
         Warning("\tdevice[%p]: Large %s (%d bytes)!  Truncating to %d!\n",
                 device, hexProps[ii].name, size, sizeof val);
      }
      printf("\tdevice[%p]: %s: 0x%llx\n",
             device, hexProps[ii].name, val);
   }
   printf("\n");

   for (ii = 0; longProps[ii].name != NULL; ii++) {
      status = clGetDeviceInfo(device, longProps[ii].param, sizeof val, &val, &size);
      if (status != CL_SUCCESS) {
         Warning("\tdevice[%p]: Unable to get %s: %s!\n",
                 device, longProps[ii].name, CLErrString(status));
         continue;
      }
      if (size > sizeof val) {
         Warning("\tdevice[%p]: Large %s (%d bytes)!  Truncating to %d!\n",
                 device, longProps[ii].name, size, sizeof val);
      }
      printf("\tdevice[%p]: %s: %lld\n",
             device, longProps[ii].name, val);
   }
}
/*
 * PrintPlatform --
 *
 *      Dumps everything about the given platform ID.
 *
 * Results:
 *      void.
 */

static void
PrintPlatform(cl_platform_id platform) {
   static struct { cl_platform_info param; const char *name; } props[] = {
      { CL_PLATFORM_PROFILE, "profile" },
      { CL_PLATFORM_VERSION, "version" },
      { CL_PLATFORM_NAME, "name" },
      { CL_PLATFORM_VENDOR, "vendor" },
      { CL_PLATFORM_EXTENSIONS, "extensions" },
      { 0, NULL },
   };
   cl_device_id *deviceList;
   cl_uint numDevices;
   cl_int status;
   char buf[65536];
   size_t size;
   int ii;

   for (ii = 0; props[ii].name != NULL; ii++) {
      status = clGetPlatformInfo(platform, props[ii].param, sizeof buf, buf, &size);
      if (status != CL_SUCCESS) {
         Warning("platform[%p]: Unable to get %s: %s\n",
                 platform, props[ii].name, CLErrString(status));
         continue;
      }
      if (size > sizeof buf) {
         Warning("platform[%p]: Huge %s (%d bytes)!  Truncating to %d\n",
                 platform, props[ii].name, size, sizeof buf);
      }
      printf("platform[%p]: %s: %s\n", platform, props[ii].name, buf);
   }

   if ((status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                0, NULL, &numDevices)) != CL_SUCCESS) {
      Warning("platform[%p]: Unable to query the number of devices: %s\n",
              platform, CLErrString(status));
      return;
   }
   printf("platform[%p]: Found %d device(s).\n", platform, numDevices);

   deviceList = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
   if ((status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                numDevices, deviceList, NULL)) != CL_SUCCESS) {
      Warning("platform[%p]: Unable to enumerate the devices: %s\n",
              platform, CLErrString(status));
      free(deviceList);
      return;
   }

   for (ii = 0; ii < numDevices; ii++) {
      PrintDevice(deviceList[ii]);
   }

   free(deviceList);
}


int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
      
    float data[DATA_SIZE];              // original data set given to device
    float results[DATA_SIZE];           // results returned from device
    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

	cl_platform_id platform;			// compute platform id
	cl_uint numPlatforms;
	cl_platform_id *platformList;
	cl_int status;

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
    // Fill our data set with random float values
    //
    int i = 0;
    unsigned int count = DATA_SIZE;
    for(i = 0; i < count; i++)
	{
        data[i] = rand() / (float)RAND_MAX;
	//	printf("%f\n",data[i]);
	}
    

	err = clGetPlatformIDs(1, &platform, &numPlatforms);            
	if(err < 0) {          
		perror("Couldn't find any platforms");
		  return EXIT_FAILURE;
	}
	printf("Found %d platform(s).\n", numPlatforms);
	
	platformList = (cl_platform_id *) malloc(sizeof(cl_platform_id) * numPlatforms);
    if ((status = clGetPlatformIDs(numPlatforms, platformList, NULL)) != CL_SUCCESS) {
       Warning("Unable to enumerate the platforms: %s\n",
               CLErrString(status));
       exit(1);
    }

	int ii;
    for (ii = 0; ii < numPlatforms; ii++) {
       PrintPlatform(platformList[ii]);
    }

    
    // Connect to a compute device
    //
    int gpu = 0;
    err = clGetDeviceIDs(platformList[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

  

    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    
    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // Validate our results
    //
    correct = 0;
    for(i = 0; i < count; i++)
    {
		//printf("%4f %4.10f %4.10f \n", data[i],data[i] * data[i], results[i]);
        if((results[i] - data[i] * data[i]) < 10e-8)
            correct++;
    }
    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values!\n", correct, count);
    
    // Shutdown and cleanup
    //
	free(platformList);
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

