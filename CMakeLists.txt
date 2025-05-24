cmake_minimum_required(VERSION 3.18)
project(Matrix LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75)
enable_language(CUDA)


add_executable(Matrix MatrixTests.cc cuda_mul.cu)
