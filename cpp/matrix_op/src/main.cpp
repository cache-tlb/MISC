#include "matrix_utils.h"
#include "timer.h"
#include "transpose.h"
#include "multiply.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct SizeSpec2D {
    int rows;
    int cols;
};

const std::vector<int> kBlockSizes = {16, 32, 64, 128};
const int kRepetitions = 5;

void print_table_header(const char* size_label) {
    std::printf("%-20s | %-9s | %13s | %14s | %10s | %s\n",
                size_label, "BlockSize", "Baseline(ms)", "Optimized(ms)", "Speedup(x)", "Correct");
    std::printf("-------------------------------------------------------------------------------------------\n");
}

void run_transpose_experiment() {
    std::printf("\n=== Experiment 1: Matrix Transpose ===\n");
    print_table_header("Size(rows x cols)");

    std::vector<SizeSpec2D> sizes = {
        {512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096}, {1000, 4100}
    };

    for (const auto& spec : sizes) {
        int rows = spec.rows;
        int cols = spec.cols;

        float* A = create_matrix(rows, cols);
        fill_matrix(A, rows, cols);
        float* B = create_matrix(cols, rows);
        float* C = create_matrix(cols, rows);

        double baseline_avg = run_benchmark([&] { transpose_baseline(A, B, rows, cols); }, kRepetitions, true);
        std::string size_label = std::to_string(rows) + " x " + std::to_string(cols);

        for (int block_size : kBlockSizes) {
            double optimized_avg = run_benchmark(
                [&] { transpose_optimized(A, C, rows, cols, block_size); }, kRepetitions, true);
            bool correct = compare_matrices(B, C, cols, rows);
            double speedup = baseline_avg / optimized_avg;

            std::printf("%-20s | %-9d | %13.3f | %14.3f | %10.2f | %s\n",
                        size_label.c_str(), block_size,
                        baseline_avg * 1000.0, optimized_avg * 1000.0,
                        speedup, correct ? "YES" : "NO");
        }

        delete_matrix(A);
        delete_matrix(B);
        delete_matrix(C);
    }
}

struct SizeSpec3D {
    int rowsA;
    int colsA;
    int colsB;
};

void run_multiply_experiment() {
    std::printf("\n=== Experiment 2: Matrix Multiply ===\n");
    print_table_header("Size(rA x cA x cB)");

    std::vector<SizeSpec3D> sizes = {
        {128, 128, 128}, {256, 256, 256}, {512, 512, 512}, {1024, 1024, 1024},
        {500, 1000, 250}
    };

    for (const auto& spec : sizes) {
        int rowsA = spec.rowsA;
        int colsA = spec.colsA;
        int colsB = spec.colsB;

        float* A = create_matrix(rowsA, colsA);
        float* B = create_matrix(colsA, colsB);
        float* C = create_matrix(rowsA, colsB);
        float* D = create_matrix(rowsA, colsB);

        fill_matrix(A, rowsA, colsA);
        fill_matrix(B, colsA, colsB);

        double baseline_avg = run_benchmark(
            [&] { multiply_baseline(A, B, C, rowsA, colsA, colsB); }, kRepetitions, true);
        std::string size_label =
            std::to_string(rowsA) + " x " + std::to_string(colsA) + " x " + std::to_string(colsB);

        for (int block_size : kBlockSizes) {
            double optimized_avg = run_benchmark(
                [&] { multiply_optimized(A, B, D, rowsA, colsA, colsB, block_size); }, kRepetitions, true);
            bool correct = compare_matrices(C, D, rowsA, colsB);
            double speedup = baseline_avg / optimized_avg;

            std::printf("%-20s | %-9d | %13.3f | %14.3f | %10.2f | %s\n",
                        size_label.c_str(), block_size,
                        baseline_avg * 1000.0, optimized_avg * 1000.0,
                        speedup, correct ? "YES" : "NO");
        }

        delete_matrix(A);
        delete_matrix(B);
        delete_matrix(C);
        delete_matrix(D);
    }
}

}  // namespace

int main() {
    run_transpose_experiment();
    run_multiply_experiment();
    return 0;
}
