#pragma once

#include <iostream>

inline int g_tests_run = 0;
inline int g_tests_failed = 0;

#define CHECK(cond) do { \
    ++g_tests_run; \
    if (!(cond)) { \
        ++g_tests_failed; \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << "  " << #cond << "\n"; \
    } \
} while (0)
