cmake_minimum_required(VERSION 2.8)
project(my_gamma)

add_executable(GammaTrans gamma_transform.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(GammaTrans ${OpenCV_LIBS})

find_package(OpenCL REQUIRED)
target_link_libraries(GammaTrans OpenCL::OpenCL)