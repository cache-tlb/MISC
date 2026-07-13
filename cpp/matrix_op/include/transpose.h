#pragma once

void transpose_baseline(const float* A, float* B, int rows, int cols);
void transpose_optimized(const float* A, float* B, int rows, int cols, int BLOCK_SIZE);
