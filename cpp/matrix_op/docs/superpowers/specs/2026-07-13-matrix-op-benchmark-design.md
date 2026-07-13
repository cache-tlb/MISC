# 矩阵运算性能对比实验框架 — 设计文档

日期：2026-07-13

## 背景与目标

实现一个 C++ 性能对比实验框架，对比两组矩阵运算（转置、乘法）的朴素实现与缓存分块优化实现的性能差异，并验证两种实现结果一致。矩阵为 `float` 类型、行主序存储（`A[i * cols + j]`）。

## 工具链

本机未检测到 g++/clang++，仅有 Visual Studio 2022 Enterprise（MSVC）与 CMake。采用 **CMake + MSVC**：

- Release 配置下映射 `-O3` 语义为 MSVC 等价的高优化选项：`/O2 /Oi /GL /fp:fast`（`/GL` 需配合链接期 `/LTCG`）。
- `CMakeLists.txt` 用 `if(MSVC)` 分支处理，若未来机器上装有 g++/clang++，则自动使用 `-O3 -march=native`，保持可移植性。
- C++ 标准：C++17。

## 项目结构

```
matrix_op/
├── CMakeLists.txt
├── include/
│   ├── matrix_utils.h    # 公共接口：create/clone/delete/fill/compare
│   ├── timer.h            # 高精度计时 + run_benchmark 通用测量工具
│   ├── transpose.h        # transpose_baseline / transpose_optimized
│   └── multiply.h         # multiply_baseline / multiply_optimized
├── src/
│   ├── matrix_utils.cpp
│   ├── transpose.cpp
│   ├── multiply.cpp
│   └── main.cpp            # 驱动两个实验，扫描规模与 BLOCK_SIZE，打印结果表
```

各模块职责单一：`matrix_utils` 只管数据生命周期与比较，不涉及算法；`transpose`/`multiply` 只管算法，不涉及计时；`main` 只负责编排实验与输出，不实现算法细节。

## 公共接口（matrix_utils.h / .cpp）

```cpp
float* create_matrix(int rows, int cols);   // 元素初始化为 0
float* clone_matrix(const float* matrix, int rows, int cols);
void   delete_matrix(float* matrix);
void   fill_matrix(float* matrix, int rows, int cols); // [-1, 1] 随机浮点数
bool   compare_matrices(const float* A, const float* B, int rows, int cols);
```

- `fill_matrix` 内部使用固定种子的 `std::mt19937`（每次调用重新播种为同一固定种子），保证同一次程序运行内数据可复现。
- `compare_matrices` 使用带容差的近似比较，而非逐位相等：

  ```cpp
  bool nearly_equal(float a, float b) {
      float diff = std::fabs(a - b);
      float scale = std::max(std::fabs(a), std::fabs(b));
      return diff <= std::max(1e-5f, 1e-4f * scale);
  }
  ```

  **原因**：分块矩阵乘法会改变浮点加法的累加顺序，理论上可能与朴素实现产生极小的浮点舍入差异，严格逐位比较不能反映"数值等价"的真实语义。转置实验只是数据搬运不涉及浮点运算，容差比较对它同样成立（差异应为 0）。

## 计时框架（timer.h）

```cpp
double run_benchmark(const std::function<void()>& func,
                      int repetitions = 5, bool warmup = true);
```

- 若 `warmup` 为真，先执行一次不计时的调用（避免首次缓存/分页开销干扰测量）。
- 随后执行 `repetitions` 次，用 `std::chrono::high_resolution_clock` 分别计时，返回算术平均值（秒）。
- 转置/乘法两个实验共用此函数，避免重复计时逻辑。
- 正确性比较使用计时循环中最后一次调用遗留在输出缓冲区中的结果（两种算法均为确定性算法，结果与运行次数无关，无需额外单独运行）。

## 实验一：矩阵转置

### 接口

```cpp
void transpose_baseline(const float* A, float* B, int rows, int cols);
void transpose_optimized(const float* A, float* B, int rows, int cols, int BLOCK_SIZE);
```

- baseline：双重循环，`B[j * rows + i] = A[i * cols + j]`。
- optimized：按 `BLOCK_SIZE × BLOCK_SIZE` 分块遍历 `(i, j)`，块内做逐元素转置，减少大跨步访问导致的缓存行浪费；块在边界处需裁剪（`std::min(ii + BLOCK_SIZE, rows)`）以支持不能整除的规模。

### 测试规模矩阵

| 规模 (rows × cols) | 说明 |
|---|---|
| 512 × 512 | |
| 1024 × 1024 | |
| 2048 × 2048 | |
| 4096 × 4096 | |
| 1000 × 4100 | 非方阵，且两维均不能被任一候选 BLOCK_SIZE（16/32/64/128）整除，用于触发分块边界裁剪逻辑 |

BLOCK_SIZE 候选：16, 32, 64, 128（对每个规模逐一测试）。

### 测试流程（沿用需求中的步骤，套入计时框架与规模/BLOCK_SIZE 双重循环）

对每个规模：
1. `A = create_matrix(rows, cols); fill_matrix(A, rows, cols);`
2. `B = clone_matrix(A, cols, rows); C = clone_matrix(A, cols, rows);`（作为输出缓冲区，形状为转置后的形状）
3. baseline 计时：`run_benchmark([&]{ transpose_baseline(A, B, rows, cols); })` → 得到平均耗时
4. 对每个 BLOCK_SIZE：optimized 计时：`run_benchmark([&]{ transpose_optimized(A, C, rows, cols, BLOCK_SIZE); })` → 得到平均耗时
5. `compare_matrices(B, C, cols, rows)` 验证正确性
6. 释放内存，打印该规模/BLOCK_SIZE 组合的结果行

## 实验二：矩阵乘法

### 接口

```cpp
void multiply_baseline(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
void multiply_optimized(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB, int BLOCK_SIZE);
```

- baseline：朴素三重循环 `ijk` 序，`C[i*colsB+j] += A[i*colsA+k] * B[k*colsB+j]`。
- optimized：分块 + 块内 `ikj` 循环序（而非朴素 `ijk`），使块内对 `B`、`C` 的访问按行连续，是行主序存储下缓存友好的标准写法：

  ```cpp
  std::fill(C, C + rowsA * colsB, 0.0f);
  for (块 ii, kk, jj 按 BLOCK_SIZE 划分，边界裁剪)
    for (i in 块) for (k in 块) {
      float a = A[i*colsA+k];
      for (j in 块) C[i*colsB+j] += a * B[k*colsB+j];
    }
  ```

### 测试规模矩阵

| rowsA | colsA | colsB | 说明 |
|---|---|---|---|
| 128 | 128 | 128 | 方阵 |
| 256 | 256 | 256 | 方阵 |
| 512 | 512 | 512 | 方阵 |
| 1024 | 1024 | 1024 | 方阵 |
| 500 | 1000 | 250 | 非方阵，三维度均不相同且均不能被任一候选 BLOCK_SIZE（16/32/64/128）整除，用于触发分块边界裁剪逻辑 |

BLOCK_SIZE 候选：16, 32, 64, 128（对每组规模逐一测试）。

由于矩阵乘法是 O(n³) 复杂度，规模上限低于转置实验以控制总运行时间。

### 测试流程

对每组规模：
1. `A = create_matrix(rowsA, colsA); B = create_matrix(colsA, colsB);`
2. `C = create_matrix(rowsA, colsB); D = create_matrix(rowsA, colsB);`
3. `fill_matrix(A, rowsA, colsA); fill_matrix(B, colsA, colsB);`
4. baseline 计时：`run_benchmark([&]{ multiply_baseline(A, B, C, rowsA, colsA, colsB); })`
5. 对每个 BLOCK_SIZE：optimized 计时：`run_benchmark([&]{ multiply_optimized(A, B, D, rowsA, colsA, colsB, BLOCK_SIZE); })`
6. `compare_matrices(C, D, rowsA, colsB)` 验证正确性
7. 释放内存，打印该规模/BLOCK_SIZE 组合的结果行

## 输出格式

控制台对齐表格，两个实验分别输出各自的表格，列为：

```
Size            | BlockSize | Baseline(ms) | Optimized(ms) | Speedup(x) | Correct
```

矩阵乘法的 `Size` 列显示为 `rowsA x colsA x colsB` 形式；转置的 `Size` 列显示为 `rows x cols`。

## 非目标（YAGNI）

- 不引入多线程/SIMD 显式优化（仅缓存分块），保持与需求范围一致。
- 不输出 CSV 或图表文件，仅控制台表格输出。
- 不做超出 `[-1,1]` 均匀分布之外的其他数据分布测试。
