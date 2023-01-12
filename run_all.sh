rm -rf build
mkdir build && cd build
cmake ..
cmake --build .
cp ../custom_opencl_kernels.cl .
./GammaTrans ../sample_images/Lawliet.jpg 2.2