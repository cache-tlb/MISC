#include "matrix_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

float* create_matrix(int rows, int cols) {
    size_t count = static_cast<size_t>(rows) * cols;
    float* matrix = new float[count];
    std::fill(matrix, matrix + count, 0.0f);
    return matrix;
}

float* clone_matrix(const float* matrix, int rows, int cols) {
    size_t count = static_cast<size_t>(rows) * cols;
    float* copy = new float[count];
    std::memcpy(copy, matrix, count * sizeof(float));
    return copy;
}

void delete_matrix(float* matrix) {
    delete[] matrix;
}

void fill_matrix(float* matrix, int rows, int cols) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t count = static_cast<size_t>(rows) * cols;
    for (size_t i = 0; i < count; ++i) {
        matrix[i] = dist(rng);
    }
}

bool compare_matrices(const float* A, const float* B, int rows, int cols) {
    size_t count = static_cast<size_t>(rows) * cols;
    for (size_t i = 0; i < count; ++i) {
        float diff = std::fabs(A[i] - B[i]);
        float scale = std::max(std::fabs(A[i]), std::fabs(B[i]));
        float tolerance = std::max(1e-5f, 1e-4f * scale);
        if (diff > tolerance) {
            return false;
        }
    }
    return true;
}
