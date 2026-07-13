#include "test_framework.h"
#include "transpose.h"
#include "matrix_utils.h"

void test_transpose_suite() {
    // transpose_baseline correctly transposes a small known matrix
    {
        float A[6] = {1, 2, 3, 4, 5, 6};  // 2x3
        float B[6] = {0};
        transpose_baseline(A, B, 2, 3);
        float expected[6] = {1, 4, 2, 5, 3, 6};  // 3x2
        CHECK(compare_matrices(B, expected, 3, 2));
    }

    // When BLOCK_SIZE divides matrix dimensions evenly, transpose_optimized matches baseline
    {
        float* A = create_matrix(64, 64);
        fill_matrix(A, 64, 64);
        float* B = create_matrix(64, 64);
        float* C = create_matrix(64, 64);
        transpose_baseline(A, B, 64, 64);
        transpose_optimized(A, C, 64, 64, 16);
        CHECK(compare_matrices(B, C, 64, 64));
        delete_matrix(A);
        delete_matrix(B);
        delete_matrix(C);
    }

    // When BLOCK_SIZE does not divide matrix dimensions (boundary clipping), results still match
    {
        float* A = create_matrix(50, 70);
        fill_matrix(A, 50, 70);
        float* B = create_matrix(70, 50);
        float* C = create_matrix(70, 50);
        transpose_baseline(A, B, 50, 70);
        transpose_optimized(A, C, 50, 70, 32);
        CHECK(compare_matrices(B, C, 70, 50));
        delete_matrix(A);
        delete_matrix(B);
        delete_matrix(C);
    }
}
