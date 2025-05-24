#include "cuda_mul.h"
#include <cuda_runtime.h>
#include <iostream>

#define HANDLE_CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(-1); \
    } \
}

__global__ void matrixMulKernel(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; ++k) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

void cudaMatrixMultiply(const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int A_rows, int A_cols, int B_cols) {
    float* d_A, * d_B, * d_C;

    size_t size_A = A_rows * A_cols * sizeof(float);
    size_t size_B = A_cols * B_cols * sizeof(float);
    size_t size_C = A_rows * B_cols * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&d_A, size_A);
    HANDLE_CUDA_ERROR(err);
    err = cudaMalloc(&d_B, size_B);
    HANDLE_CUDA_ERROR(err);
    err = cudaMalloc(&d_C, size_C);
    HANDLE_CUDA_ERROR(err);

    err = cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(err);
    err = cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(err);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((B_cols + 31) / 32, (A_rows + 31) / 32);

    matrixMulKernel << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, A_rows, A_cols, B_cols);
    err = cudaGetLastError();
    HANDLE_CUDA_ERROR(err);

    err = cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost);
    HANDLE_CUDA_ERROR(err);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
