#pragma once

float* create_matrix(int rows, int cols);
float* clone_matrix(const float* matrix, int rows, int cols);
void delete_matrix(float* matrix);
void fill_matrix(float* matrix, int rows, int cols);
bool compare_matrices(const float* A, const float* B, int rows, int cols);
