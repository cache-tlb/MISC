#include "transpose.h"

#include <algorithm>

void transpose_baseline(const float* A, float* B, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

void transpose_optimized(const float* A, float* B, int rows, int cols, int BLOCK_SIZE) {
    for (int ii = 0; ii < rows; ii += BLOCK_SIZE) {
        int iMax = std::min(ii + BLOCK_SIZE, rows);
        for (int jj = 0; jj < cols; jj += BLOCK_SIZE) {
            int jMax = std::min(jj + BLOCK_SIZE, cols);
            for (int i = ii; i < iMax; ++i) {
                for (int j = jj; j < jMax; ++j) {
                    B[j * rows + i] = A[i * cols + j];
                }
            }
        }
    }
}
