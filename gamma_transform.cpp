#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#define MAX_SOURCE_SIZE (0x100000)

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main(int argc, char** argv) {

    // check and load image using opencv api
    if (argc != 3) {
        printf("Usage: img_proc <image_path> <gamma_value> \n");
        return -1;
    }

    /* Image and LUT preparation */

    cv::Mat img;
    img = cv::imread(argv[1], 1);
    if (!img.data) {
        printf("No image data found! \n");
        return -1;
    }

    // convert to grayscale for simplicity of processing as example
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    // display original gray image
    const char * win_name_ori = "original image gray";
    cv::namedWindow(win_name_ori, cv::WINDOW_AUTOSIZE);
    cv::imshow(win_name_ori, img_gray);
    // cv::waitKey(0);

    int nrows = img_gray.rows;
    int ncols = img_gray.cols;
    int img_buffer_size = nrows * ncols * 3 * sizeof(char);
    uchar * buffer_src  = (uchar *)malloc(img_buffer_size);
    uchar * buffer_dst  = (uchar *)malloc(img_buffer_size);
    memcpy(buffer_src, img_gray.data, img_buffer_size);
    memset(buffer_dst, 0x00, img_buffer_size);

    // calculate LUT for gamma tranform given GAMMA
    const double GAMMA = atof(argv[2]);
    uchar * lut_255  = (uchar *)malloc(256);
    for (int i = 0; i <= 255; i++) {
        lut_255[i] = (uchar)(pow((double)i / 255.0, GAMMA) * 255.0);
    }

    /* OpenCL preparation */

    std::cout << "OpenCL Max Memory alloc: " << CL_DEVICE_MAX_MEM_ALLOC_SIZE << std::endl;
    std::cout << "OpenCL Min Align size: " << CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE << std::endl;
    
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("custom_opencl_kernels.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // define opencl env variables
    cl_int error = 0;
    cl_int cur_error = 0;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_context context;
    cl_command_queue command_queue;

    // get platform
    error |= clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (error != CL_SUCCESS) {
        std::cerr << "Error in get platform ids: " << std::to_string(error) << std::endl;
        exit(error);
    }
    std::cout << "Number of platforms: " << std::to_string(ret_num_platforms) << std::endl;

    // get device
    error |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
    if (error != CL_SUCCESS) {
        std::cerr << "Error in get device ids: " << std::to_string(error) << std::endl;
        exit(error);
    }

    // get context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &cur_error);
    if (cur_error != CL_SUCCESS) {
        std::cerr << "Error in get context: " << std::to_string(cur_error) << std::endl;
        exit(error);
    }

    // Command-queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &cur_error);
    if (cur_error != CL_SUCCESS) {
        std::cerr << "Error creating command queue: " << std::to_string(cur_error) << std::endl;
        exit(error);
    }

    cl_mem mem_buffer_src = clCreateBuffer(context, CL_MEM_READ_ONLY, img_buffer_size, NULL, &cur_error);
    cl_mem mem_buffer_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, img_buffer_size, NULL, &cur_error);
    cl_mem mem_buffer_lut = clCreateBuffer(context, CL_MEM_READ_ONLY, 256, NULL, &cur_error);
    if (cur_error != CL_SUCCESS) {
        std::cerr << "Error creating CL buffer: " << std::to_string(cur_error) << std::endl;
        exit(error);
    }

    error |= clEnqueueWriteBuffer(command_queue, mem_buffer_src, CL_TRUE, 0, img_buffer_size, buffer_src, 0, NULL, NULL);
    error |= clEnqueueWriteBuffer(command_queue, mem_buffer_dst, CL_TRUE, 0, img_buffer_size, buffer_dst, 0, NULL, NULL);
    error |= clEnqueueWriteBuffer(command_queue, mem_buffer_lut, CL_TRUE, 0, 256, lut_255, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        std::cerr << "Error in enqueue buffer: " << std::to_string(error) << std::endl;
        exit(error);
    }

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &cur_error);
    if (cur_error != CL_SUCCESS) {
        std::cerr << "Error in create program: " << std::to_string(cur_error) << std::endl;
        exit(cur_error);
    }

    // Build the program
    error |= clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (error != CL_SUCCESS) {
        std::cerr << "Error in build program: " << std::to_string(error) << std::endl;
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        // Allocate memory for the log
        char *log = (char *) malloc(log_size);
        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        // Print the log
        printf("%s\n", log);
        exit(error);
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "gamma_transform", &cur_error);

    // Set the arguments of the kernel
    error |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (uchar *)&mem_buffer_src);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (uchar *)&mem_buffer_dst);
    error |= clSetKernelArg(kernel, 2, sizeof(cl_int), &ncols);
    error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &mem_buffer_lut);

    if (error != CL_SUCCESS) {
        std::cerr << "Error in set kernel: " << std::to_string(error) << std::endl;
        exit(error);
    }

    size_t global_work_size[2] = {(size_t)nrows, (size_t)ncols};
    size_t local_work_size[2] = {1, 1};

    error |= clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
            global_work_size, local_work_size, 0, NULL, NULL);

    if (error != CL_SUCCESS) {
        std::cerr << "Error in kernel enqueue: " << std::to_string(error) << std::endl;
        exit(error);
    }

    // Read the memory buffer dst on the device to the local variable dst
    error |= clEnqueueReadBuffer(command_queue, mem_buffer_dst, CL_TRUE, 0, 
            img_buffer_size, buffer_dst, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        std::cerr << "Error in read buffer: " << std::to_string(error) << std::endl;
        exit(error);
    }

    // read processed output as cv::Mat and display
    // cv::Mat img_gray_out = img_gray.clone();
    // memcpy(img_gray_out.data, buffer_dst, img_buffer_size);
    cv::Mat img_gray_out(img_gray.rows, img_gray.cols, img_gray.type(), buffer_dst);
    
    // display processed gray image
    const char * win_name_out = "processed image gray";
    cv::namedWindow(win_name_out, cv::WINDOW_AUTOSIZE);
    cv::imshow(win_name_out, img_gray_out);
    cv::waitKey(0);

    // Clean up
    error = clFlush(command_queue);
    error = clFinish(command_queue);
    error = clReleaseKernel(kernel);
    error = clReleaseProgram(program);
    error = clReleaseMemObject(mem_buffer_src);
    error = clReleaseMemObject(mem_buffer_dst);
    error = clReleaseCommandQueue(command_queue);
    error = clReleaseContext(context);
    free(buffer_src);
    free(buffer_dst);

    return 0;
}