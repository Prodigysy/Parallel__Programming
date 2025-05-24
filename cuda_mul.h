#pragma once
#include <vector>

void cudaMatrixMultiply(const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int A_rows, int A_cols, int B_cols);
