//------------------------------------------------------------------------------
//
// Name:       vadd_chain.cpp
//
// Purpose:    Elementwise addition of two vectors at a time in a chain (C=A+B; D=C+E; F=D+G)
//
// HISTORY:    Initial version based on vadd.c, written by Tim Mattson, June 2011
//             Ported to C++ Wrapper API by Benedict Gaster, September 2011
//             Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Ported back to C by Tom Deakin, July 2013
//             Updated by Tom Deakin, October 2014
//
//------------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"
#include "cache.h"

//pick up device type from compiler command line or from
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern int output_device_info(cl_device_id );

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (10)    // length of vectors a, b, and c

//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Compute the elementwise sum c = a+b
//
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//

const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";

//------------------------------------------------------------------------------
const int UseCache = 0;



int main(int argc, char** argv)
{
    cl_int          err;               // error code returned from OpenCL calls

    size_t dataSize = sizeof(int) * LENGTH;
    int*       h_a = (int *)malloc(dataSize);       // a vector
    int*       h_b = (int *)malloc(dataSize);       // b vector
    int*       h_c = (int *)malloc(dataSize);       // c vector (result)
    int*       h_d = (int *)malloc(dataSize);       // d vector (result)
    int*       h_e = (int *)malloc(dataSize);       // e vector
    int*       h_f = (int *)malloc(dataSize);       // f vector (result)
    int*       h_g = (int *)malloc(dataSize);       // g vector
    unsigned int correct;           // number of correct results

    size_t global;                  // global domain size

    cl_device_id     device_id;     // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        ko_vadd;       // compute kernel

    //-----------------------------------------------------
    //------------------ Create Cache ---------------------

    struct Cache_t* myCache = CreateCache(8, dataSize, 32, direct_mapped, random_RP);

    cl_mem input1;
    cl_mem input2;
    cl_mem output;
    //-----------------------------------------------------

    // Fill vectors a and b with random float values
    int i = 0;
    for(i = 0; i < LENGTH; i++){
        h_a[i] = rand()%100;
        h_b[i] = rand()%100;
        h_d[i] = rand()%100;
        h_f[i] = rand()%100;
    }

    // Set up platform and GPU device

    cl_uint numPlatforms;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    for (i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    if (device_id == NULL)
        checkError(err, "Getting device");

    err = output_device_info(device_id);
    checkError(err, "Outputting device info");
  
    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    checkError(err, "Creating program");

    // Build the program  
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program 
    ko_vadd = clCreateKernel(program, "vadd", &err);
    checkError(err, "Creating kernel");
    const int count = LENGTH;


    //-----------------------------------------------------
    //-----------------Write to cache----------------------

    input1 = clCreateCacheBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dataSize, h_a, &err, myCache);
    input2 = clCreateCacheBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dataSize, h_b, &err, myCache);
    output = clCreateCacheBuffer(context, CL_MEM_READ_WRITE, dataSize, h_c, &err, myCache);
    //-----------------------------------------------------

    // Enqueue kernel - first time
    // Set the arguments to the compute kernel
    err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &input1);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &input2);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
    checkError(err, "Setting kernel arguments"); 

    // Execute the kernel over the entire range of the 1d input data set
    // letting the OpenCL runtime choose the work-group size
    global = count;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel 1st time");

   //Read back the result
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(int) * count, h_c, 0, NULL, NULL );  
    checkError(err, "Reading back h_c");

    input1 = clCreateCacheBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dataSize, h_c, &err, myCache);
    input2 = clCreateCacheBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dataSize, h_d, &err, myCache);
    output = clCreateCacheBuffer(context, CL_MEM_READ_WRITE, dataSize, h_e, &err, myCache);
    // Enqueue kernel - second time
    // Set different arguments to the compute kernel
    err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &input1);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &input2);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &output);
    checkError(err, "Setting kernel arguments");
    
    // Enqueue the kernel again    
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel 2nd time");

   //Read back the result
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(int) * count, h_e, 0, NULL, NULL );  
    checkError(err, "Reading back h_e");

    input1 = clCreateCacheBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dataSize, h_e, &err, myCache);
    input2 = clCreateCacheBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dataSize, h_f, &err, myCache);
    output = clCreateCacheBuffer(context, CL_MEM_READ_WRITE, dataSize, h_g, &err, myCache);
    // Enqueue kernel - third time
    // Set different (again) arguments to the compute kernel
    err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &input1);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &input2);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &output);
    checkError(err, "Setting kernel arguments");

    // Enqueue the kernel again    
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel 3rd time");

    // Read back the result from the compute device
    err = clEnqueueReadCacheBuffer(commands, CL_TRUE, 0, sizeof(int) * count, h_g, 0, NULL, NULL, myCache); 
    checkError(err, "Reading back h_g with Get()");

    // Test the results
    //-----------------------------------------------------
    correct = 0;
    int tmp;
    
    for(i = 0; i < count; i++)
    {
        tmp = h_f[i]+(h_d[i]+(h_a[i] + h_b[i]));
        tmp -= h_g[i];                               
        if(tmp == 0)                        // correct if square deviation is less than tolerance squared
            correct++;
        else {
            printf("h_a=%d h_b=%d h_c=%d h_d=%d h_e=%d h_f=%d h_g=%d\n", h_a[i], h_b[i], h_c[i], h_d[i], h_e[i], h_f[i],h_g[i]);
        }
    }

    // summarize results
    printf("G = A+B+D+F:  %d out of %d results were correct.\n", correct, count);
    printf("Number of transfers via Cache = %d, WriteTransfers = %d\n",myCache->memCopies, myCache->WriteTransfers);
    if(correct == count){
        printf("\n-----------------------------------------------\n");
        printf("-----------------Test Passed!------------------\n");
        printf("-----------------------------------------------\n\n");
    }else{
        printf("\n-----------------------------------------------\n");
        printf("-----------------Test Failed!------------------\n");
        printf("-----------------------------------------------\n\n");
    }

    //-----------------------------------------------------
    //-------------------- Free Cache ---------------------

    FreeCache(myCache);
    
   //-----------------------------------------------------

    // cleanup then shutdown
    clReleaseProgram(program);
    clReleaseKernel(ko_vadd);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(h_e);
    free(h_f);
    free(h_g);

    return 0;
}

