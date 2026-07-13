#include "test_framework.h"

#include <iostream>

void test_matrix_utils_suite();
void test_timer_suite();
void test_transpose_suite();
void test_multiply_suite();

int main() {
    test_matrix_utils_suite();
    test_timer_suite();
    test_transpose_suite();
    test_multiply_suite();

    std::cout << "\n" << g_tests_run << " checks run, " << g_tests_failed << " failed.\n";
    return g_tests_failed == 0 ? 0 : 1;
}
