#pragma once

#include <chrono>
#include <functional>

inline double run_benchmark(const std::function<void()>& func, int repetitions = 5, bool warmup = true) {
    if (warmup) {
        func();
    }
    double total_seconds = 0.0;
    for (int i = 0; i < repetitions; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        total_seconds += std::chrono::duration<double>(end - start).count();
    }
    return total_seconds / repetitions;
}
