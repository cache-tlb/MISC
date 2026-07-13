#include "multiply.h"

#include <algorithm>
#include <cstring>

void multiply_baseline(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
}

void multiply_optimized(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB, int BLOCK_SIZE) {
    std::memset(C, 0, static_cast<size_t>(rowsA) * colsB * sizeof(float));

    for (int ii = 0; ii < rowsA; ii += BLOCK_SIZE) {
        int iMax = std::min(ii + BLOCK_SIZE, rowsA);
        for (int kk = 0; kk < colsA; kk += BLOCK_SIZE) {
            int kMax = std::min(kk + BLOCK_SIZE, colsA);
            for (int jj = 0; jj < colsB; jj += BLOCK_SIZE) {
                int jMax = std::min(jj + BLOCK_SIZE, colsB);
                for (int i = ii; i < iMax; ++i) {
                    for (int k = kk; k < kMax; ++k) {
                        float a = A[i * colsA + k];
                        for (int j = jj; j < jMax; ++j) {
                            C[i * colsB + j] += a * B[k * colsB + j];
                        }
                    }
                }
            }
        }
    }
}
