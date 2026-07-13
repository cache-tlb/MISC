#pragma once

void multiply_baseline(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
void multiply_optimized(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB, int BLOCK_SIZE);
