#include "test_framework.h"
#include "multiply.h"
#include "matrix_utils.h"

void test_multiply_suite() {
    // multiply_baseline computes correctly on a small known matrix
    // A(2x3) = [[1,2,3],[4,5,6]], B(3x2) = [[7,8],[9,10],[11,12]]
    // C = A*B = [[58,64],[139,154]]
    {
        float A[6] = {1, 2, 3, 4, 5, 6};
        float B[6] = {7, 8, 9, 10, 11, 12};
        float C[4] = {0};
        multiply_baseline(A, B, C, 2, 3, 2);
        float expected[4] = {58, 64, 139, 154};
        CHECK(compare_matrices(C, expected, 2, 2));
    }

    // When BLOCK_SIZE divides matrix dimensions evenly, multiply_optimized matches baseline on square matrices
    {
        float* A = create_matrix(64, 64);
        float* B = create_matrix(64, 64);
        fill_matrix(A, 64, 64);
        fill_matrix(B, 64, 64);
        float* C = create_matrix(64, 64);
        float* D = create_matrix(64, 64);
        multiply_baseline(A, B, C, 64, 64, 64);
        multiply_optimized(A, B, D, 64, 64, 64, 16);
        CHECK(compare_matrices(C, D, 64, 64));
        delete_matrix(A);
        delete_matrix(B);
        delete_matrix(C);
        delete_matrix(D);
    }

    // When BLOCK_SIZE does not divide matrix dimensions (boundary clipping), results still match
    {
        float* A = create_matrix(50, 70);
        float* B = create_matrix(70, 40);
        fill_matrix(A, 50, 70);
        fill_matrix(B, 70, 40);
        float* C = create_matrix(50, 40);
        float* D = create_matrix(50, 40);
        multiply_baseline(A, B, C, 50, 70, 40);
        multiply_optimized(A, B, D, 50, 70, 40, 32);
        CHECK(compare_matrices(C, D, 50, 40));
        delete_matrix(A);
        delete_matrix(B);
        delete_matrix(C);
        delete_matrix(D);
    }
}
