#include "test_framework.h"
#include "matrix_utils.h"

void test_matrix_utils_suite() {
    // create_matrix returns zero-initialized buffer
    {
        float* m = create_matrix(3, 4);
        bool all_zero = true;
        for (int i = 0; i < 12; ++i) {
            if (m[i] != 0.0f) all_zero = false;
        }
        CHECK(all_zero);
        delete_matrix(m);
    }

    // clone_matrix copies values and is independent from original buffer
    {
        float* m = create_matrix(2, 2);
        m[0] = 1.0f; m[1] = 2.0f; m[2] = 3.0f; m[3] = 4.0f;
        float* c = clone_matrix(m, 2, 2);
        CHECK(compare_matrices(m, c, 2, 2));
        c[0] = 99.0f;
        CHECK(m[0] == 1.0f);
        delete_matrix(m);
        delete_matrix(c);
    }

    // fill_matrix generates values in [-1, 1] and not all zeros
    {
        float* m = create_matrix(10, 10);
        fill_matrix(m, 10, 10);
        bool in_range = true;
        bool has_nonzero = false;
        for (int i = 0; i < 100; ++i) {
            if (m[i] < -1.0f || m[i] > 1.0f) in_range = false;
            if (m[i] != 0.0f) has_nonzero = true;
        }
        CHECK(in_range);
        CHECK(has_nonzero);
        delete_matrix(m);
    }

    // fill_matrix does not reset random stream on each call: consecutive calls give different results
    {
        float* m1 = create_matrix(5, 5);
        float* m2 = create_matrix(5, 5);
        fill_matrix(m1, 5, 5);
        fill_matrix(m2, 5, 5);
        CHECK(!compare_matrices(m1, m2, 5, 5));
        delete_matrix(m1);
        delete_matrix(m2);
    }

    // compare_matrices detects differences beyond tolerance
    {
        float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float b[4] = {1.0f, 2.0f, 3.0f, 4.5f};
        CHECK(!compare_matrices(a, b, 2, 2));
    }

    // compare_matrices accepts small floating point errors within tolerance
    {
        float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float b[4] = {1.0f, 2.0f, 3.0f, 4.0f + 1e-6f};
        CHECK(compare_matrices(a, b, 2, 2));
    }
}