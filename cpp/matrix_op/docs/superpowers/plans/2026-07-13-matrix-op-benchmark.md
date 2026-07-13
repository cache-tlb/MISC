# 矩阵运算性能对比实验框架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个 C++ 性能对比实验框架，对比矩阵转置、矩阵乘法的朴素实现与缓存分块优化实现的性能与正确性。

**Architecture:** CMake 构建出一个静态库 `matrix_op_lib`（矩阵工具函数 + 转置/乘法算法）、一个基准测试可执行文件 `matrix_bench`（驱动两组实验、打印结果表）、一个单元测试可执行文件 `matrix_tests`（自定义轻量断言框架，逐个验证每个函数的正确性）。

**Tech Stack:** C++17，CMake ≥ 3.15，MSVC（Visual Studio 2022 生成器，本机已验证可用；若未来环境有 g++/clang++ 会自动切换到 `-O3 -march=native`）。

## Global Constraints

- 矩阵元素类型固定为 `float`，行主序存储：`A[i * cols + j]`。
- `fill_matrix` 填充范围为 `[-1, 1]`。
- Release 构建下 MSVC 使用 `/O2 /Oi /GL /fp:fast`（+ 链接期 `/LTCG`）作为 `-O3` 的等价映射；非 MSVC 编译器使用 `-O3 -march=native`。
- `compare_matrices` 使用带容差的近似比较：`diff <= max(1e-5f, 1e-4f * max(|a|, |b|))`，因为分块乘法会改变浮点累加顺序。
- 计时统一通过 `run_benchmark(func, repetitions=5, warmup=true)`：1 次预热（不计时）+ 5 次正式计时取平均值（秒）。
- BLOCK_SIZE 候选集固定为 `{16, 32, 64, 128}`，对每个测试规模逐一测试。
- 转置测试规模：512², 1024², 2048², 4096², 1000×4100（非方阵，故意不能被任一 BLOCK_SIZE 整除）。
- 乘法测试规模：128², 256², 512², 1024²（方阵），500×1000×250（非方阵，三维均不能被任一 BLOCK_SIZE 整除）。
- 工具链已验证：`cmake -S . -B build` 自动选中 `Visual Studio 17 2022` 生成器和 MSVC；`cmake --build build --config Release` 产出的可执行文件位于 `build/Release/<target>.exe`。
- 设计文档：`docs/superpowers/specs/2026-07-13-matrix-op-benchmark-design.md`（本计划的所有决策以此为准）。

---

### Task 1: 项目脚手架 — CMake 构建打通，空跑通过

**Files:**
- Create: `CMakeLists.txt`
- Create: `src/main.cpp`
- Create: `tests/test_framework.h`
- Create: `tests/test_main.cpp`

**Interfaces:**
- Produces: `matrix_bench`（可执行文件目标）、`matrix_tests`（可执行文件目标）、`CHECK(cond)` 宏（后续所有测试文件使用）、全局计数器 `g_tests_run` / `g_tests_failed`。

- [ ] **Step 1: 编写 `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.15)
project(matrix_op CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/Oi> $<$<CONFIG:Release>:/GL> $<$<CONFIG:Release>:/fp:fast>)
    add_link_options($<$<CONFIG:Release>:/LTCG>)
else()
    add_compile_options($<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-march=native>)
endif()

include_directories(include)

add_executable(matrix_bench src/main.cpp)

add_executable(matrix_tests tests/test_main.cpp)
target_include_directories(matrix_tests PRIVATE tests)

enable_testing()
add_test(NAME unit_tests COMMAND matrix_tests)
```

- [ ] **Step 2: 编写 `src/main.cpp`（占位）**

```cpp
#include <cstdio>

int main() {
    std::printf("matrix_op scaffold OK\n");
    return 0;
}
```

- [ ] **Step 3: 编写 `tests/test_framework.h`**

```cpp
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
```

- [ ] **Step 4: 编写 `tests/test_main.cpp`（暂无测试套件）**

```cpp
#include "test_framework.h"

#include <iostream>

int main() {
    std::cout << "\n" << g_tests_run << " checks run, " << g_tests_failed << " failed.\n";
    return g_tests_failed == 0 ? 0 : 1;
}
```

- [ ] **Step 5: 配置并构建**

Run (PowerShell, 项目根目录 `d:\tmp\matrix_op`):
```powershell
cmake -S . -B build
cmake --build build --config Release
```
Expected: 两个目标均编译成功，无错误。

- [ ] **Step 6: 运行验证**

Run:
```powershell
.\build\Release\matrix_bench.exe
.\build\Release\matrix_tests.exe
```
Expected:
```
matrix_op scaffold OK
```
```
0 checks run, 0 failed.
```
`matrix_tests.exe` 退出码应为 0（`$LASTEXITCODE` 为 0）。

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/main.cpp tests/test_framework.h tests/test_main.cpp
git commit -m "Add CMake scaffold with empty test runner"
```

---

### Task 2: matrix_utils — 公共矩阵接口（TDD）

**Files:**
- Create: `include/matrix_utils.h`
- Create: `src/matrix_utils.cpp`
- Create: `tests/test_matrix_utils.cpp`
- Modify: `tests/test_main.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `CHECK(cond)`、`g_tests_run`、`g_tests_failed`（来自 Task 1 的 `tests/test_framework.h`）。
- Produces:
  ```cpp
  float* create_matrix(int rows, int cols);
  float* clone_matrix(const float* matrix, int rows, int cols);
  void   delete_matrix(float* matrix);
  void   fill_matrix(float* matrix, int rows, int cols);
  bool   compare_matrices(const float* A, const float* B, int rows, int cols);
  ```
  后续所有任务（transpose、multiply、main.cpp、其余测试文件）都依赖这五个函数的签名。

- [ ] **Step 1: 编写 `include/matrix_utils.h`**

```cpp
#pragma once

float* create_matrix(int rows, int cols);
float* clone_matrix(const float* matrix, int rows, int cols);
void delete_matrix(float* matrix);
void fill_matrix(float* matrix, int rows, int cols);
bool compare_matrices(const float* A, const float* B, int rows, int cols);
```

- [ ] **Step 2: 编写失败测试 `tests/test_matrix_utils.cpp`**

```cpp
#include "test_framework.h"
#include "matrix_utils.h"

void test_matrix_utils_suite() {
    // create_matrix 返回全零初始化的缓冲区
    {
        float* m = create_matrix(3, 4);
        bool all_zero = true;
        for (int i = 0; i < 12; ++i) {
            if (m[i] != 0.0f) all_zero = false;
        }
        CHECK(all_zero);
        delete_matrix(m);
    }

    // clone_matrix 复制数值且与原缓冲区相互独立
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

    // fill_matrix 生成的数值落在 [-1, 1] 且不全为零
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

    // fill_matrix 不会在每次调用时重置随机流：连续两次调用结果不同
    {
        float* m1 = create_matrix(5, 5);
        float* m2 = create_matrix(5, 5);
        fill_matrix(m1, 5, 5);
        fill_matrix(m2, 5, 5);
        CHECK(!compare_matrices(m1, m2, 5, 5));
        delete_matrix(m1);
        delete_matrix(m2);
    }

    // compare_matrices 能检测出超出容差的差异
    {
        float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float b[4] = {1.0f, 2.0f, 3.0f, 4.5f};
        CHECK(!compare_matrices(a, b, 2, 2));
    }

    // compare_matrices 接受容差范围内的微小浮点误差
    {
        float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float b[4] = {1.0f, 2.0f, 3.0f, 4.0f + 1e-6f};
        CHECK(compare_matrices(a, b, 2, 2));
    }
}
```

- [ ] **Step 3: 注册测试套件到 `tests/test_main.cpp`**

```cpp
#include "test_framework.h"

#include <iostream>

void test_matrix_utils_suite();

int main() {
    test_matrix_utils_suite();

    std::cout << "\n" << g_tests_run << " checks run, " << g_tests_failed << " failed.\n";
    return g_tests_failed == 0 ? 0 : 1;
}
```

- [ ] **Step 4: 更新 `CMakeLists.txt`（新增库目标，接入测试文件）**

```cmake
cmake_minimum_required(VERSION 3.15)
project(matrix_op CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/Oi> $<$<CONFIG:Release>:/GL> $<$<CONFIG:Release>:/fp:fast>)
    add_link_options($<$<CONFIG:Release>:/LTCG>)
else()
    add_compile_options($<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-march=native>)
endif()

include_directories(include)

add_library(matrix_op_lib
    src/matrix_utils.cpp
)

add_executable(matrix_bench src/main.cpp)
target_link_libraries(matrix_bench PRIVATE matrix_op_lib)

add_executable(matrix_tests
    tests/test_main.cpp
    tests/test_matrix_utils.cpp
)
target_include_directories(matrix_tests PRIVATE tests)
target_link_libraries(matrix_tests PRIVATE matrix_op_lib)

enable_testing()
add_test(NAME unit_tests COMMAND matrix_tests)
```

- [ ] **Step 5: 配置并构建，确认因缺少实现而链接失败**

Run:
```powershell
cmake -S . -B build
cmake --build build --config Release
```
Expected: 链接错误（`unresolved external symbol`），因为 `src/matrix_utils.cpp` 尚不存在函数实现。

- [ ] **Step 6: 编写 `src/matrix_utils.cpp`**

```cpp
#include "matrix_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

float* create_matrix(int rows, int cols) {
    size_t count = static_cast<size_t>(rows) * cols;
    float* matrix = new float[count];
    std::fill(matrix, matrix + count, 0.0f);
    return matrix;
}

float* clone_matrix(const float* matrix, int rows, int cols) {
    size_t count = static_cast<size_t>(rows) * cols;
    float* copy = new float[count];
    std::memcpy(copy, matrix, count * sizeof(float));
    return copy;
}

void delete_matrix(float* matrix) {
    delete[] matrix;
}

void fill_matrix(float* matrix, int rows, int cols) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t count = static_cast<size_t>(rows) * cols;
    for (size_t i = 0; i < count; ++i) {
        matrix[i] = dist(rng);
    }
}

bool compare_matrices(const float* A, const float* B, int rows, int cols) {
    size_t count = static_cast<size_t>(rows) * cols;
    for (size_t i = 0; i < count; ++i) {
        float diff = std::fabs(A[i] - B[i]);
        float scale = std::max(std::fabs(A[i]), std::fabs(B[i]));
        float tolerance = std::max(1e-5f, 1e-4f * scale);
        if (diff > tolerance) {
            return false;
        }
    }
    return true;
}
```

`fill_matrix` 用函数局部 `static` 的 `std::mt19937` 只播种一次（程序运行期间持续推进随机流），而不是每次调用都重新播种为固定值——这样同一次程序运行中先后调用 `fill_matrix(A, ...)` 再 `fill_matrix(B, ...)` 会得到不同的数值（避免两矩阵内容重叠导致乘法测试掩盖 A/B 用反的 bug），同时因为程序每次运行时函数调用顺序固定，结果仍然可复现。

- [ ] **Step 7: 重新构建，确认测试通过**

Run:
```powershell
cmake --build build --config Release
.\build\Release\matrix_tests.exe
```
Expected: 输出形如 `N checks run, 0 failed.`（N ≥ 10），退出码 0。

- [ ] **Step 8: Commit**

```bash
git add include/matrix_utils.h src/matrix_utils.cpp tests/test_matrix_utils.cpp tests/test_main.cpp CMakeLists.txt
git commit -m "Implement matrix_utils with tolerance-based comparison"
```

---

### Task 3: timer.h — 通用基准测量工具（TDD）

**Files:**
- Create: `include/timer.h`
- Create: `tests/test_timer.cpp`
- Modify: `tests/test_main.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces:
  ```cpp
  double run_benchmark(const std::function<void()>& func, int repetitions = 5, bool warmup = true);
  ```
  Task 6、Task 7 的 `main.cpp` 将用它包裹 `transpose_baseline` / `transpose_optimized` / `multiply_baseline` / `multiply_optimized` 的调用来计时。

- [ ] **Step 1: 编写失败测试 `tests/test_timer.cpp`**

```cpp
#include "test_framework.h"
#include "timer.h"

void test_timer_suite() {
    // run_benchmark 返回一个大于零的平均耗时（秒）
    {
        double avg = run_benchmark([] {
            volatile long sum = 0;
            for (int i = 0; i < 100000; ++i) {
                sum += i;
            }
        }, 3, true);
        CHECK(avg > 0.0);
    }

    // 带预热时，总调用次数 = 1(预热) + repetitions
    {
        int call_count = 0;
        run_benchmark([&] { ++call_count; }, 4, true);
        CHECK(call_count == 5);
    }

    // 不带预热时，总调用次数 = repetitions
    {
        int call_count = 0;
        run_benchmark([&] { ++call_count; }, 4, false);
        CHECK(call_count == 4);
    }
}
```

- [ ] **Step 2: 注册测试套件到 `tests/test_main.cpp`**

```cpp
#include "test_framework.h"

#include <iostream>

void test_matrix_utils_suite();
void test_timer_suite();

int main() {
    test_matrix_utils_suite();
    test_timer_suite();

    std::cout << "\n" << g_tests_run << " checks run, " << g_tests_failed << " failed.\n";
    return g_tests_failed == 0 ? 0 : 1;
}
```

- [ ] **Step 3: 更新 `CMakeLists.txt`（`matrix_tests` 增加 `tests/test_timer.cpp`）**

在 `add_executable(matrix_tests ...)` 中追加一行 `tests/test_timer.cpp`：

```cmake
add_executable(matrix_tests
    tests/test_main.cpp
    tests/test_matrix_utils.cpp
    tests/test_timer.cpp
)
```

- [ ] **Step 4: 构建，确认因缺少 `include/timer.h` 而编译失败**

Run:
```powershell
cmake -S . -B build
cmake --build build --config Release
```
Expected: 编译错误，`timer.h: No such file or directory`（或 MSVC 对应的 `C1083`）。

- [ ] **Step 5: 编写 `include/timer.h`**

```cpp
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
```

- [ ] **Step 6: 重新构建，确认测试通过**

Run:
```powershell
cmake --build build --config Release
.\build\Release\matrix_tests.exe
```
Expected: `N checks run, 0 failed.`（N 比 Task 2 结束时多 3），退出码 0。

- [ ] **Step 7: Commit**

```bash
git add include/timer.h tests/test_timer.cpp tests/test_main.cpp CMakeLists.txt
git commit -m "Add run_benchmark timing harness with warmup support"
```

---

### Task 4: transpose — 基准与分块优化实现（TDD）

**Files:**
- Create: `include/transpose.h`
- Create: `src/transpose.cpp`
- Create: `tests/test_transpose.cpp`
- Modify: `tests/test_main.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `create_matrix`, `delete_matrix`, `fill_matrix`, `compare_matrices`（Task 2）。
- Produces:
  ```cpp
  void transpose_baseline(const float* A, float* B, int rows, int cols);
  void transpose_optimized(const float* A, float* B, int rows, int cols, int BLOCK_SIZE);
  ```
  Task 6 的 `main.cpp` 直接调用这两个函数。

- [ ] **Step 1: 编写 `include/transpose.h`**

```cpp
#pragma once

void transpose_baseline(const float* A, float* B, int rows, int cols);
void transpose_optimized(const float* A, float* B, int rows, int cols, int BLOCK_SIZE);
```

- [ ] **Step 2: 编写失败测试 `tests/test_transpose.cpp`**

```cpp
#include "test_framework.h"
#include "transpose.h"
#include "matrix_utils.h"

void test_transpose_suite() {
    // transpose_baseline 对已知小矩阵转置正确
    {
        float A[6] = {1, 2, 3, 4, 5, 6};  // 2x3
        float B[6] = {0};
        transpose_baseline(A, B, 2, 3);
        float expected[6] = {1, 4, 2, 5, 3, 6};  // 3x2
        CHECK(compare_matrices(B, expected, 3, 2));
    }

    // BLOCK_SIZE 能整除矩阵维度时，transpose_optimized 与 baseline 结果一致
    {
        float* A = create_matrix(64, 64);
        fill_matrix(A, 64, 64);
        float* B = create_matrix(64, 64);
        float* C = create_matrix(64, 64);
        transpose_baseline(A, B, 64, 64);
        transpose_optimized(A, C, 64, 64, 16);
        CHECK(compare_matrices(B, C, 64, 64));
        delete_matrix(A);
        delete_matrix(B);
        delete_matrix(C);
    }

    // BLOCK_SIZE 不能整除矩阵维度时（触发边界裁剪），结果依然一致
    {
        float* A = create_matrix(50, 70);
        fill_matrix(A, 50, 70);
        float* B = create_matrix(70, 50);
        float* C = create_matrix(70, 50);
        transpose_baseline(A, B, 50, 70);
        transpose_optimized(A, C, 50, 70, 32);
        CHECK(compare_matrices(B, C, 70, 50));
        delete_matrix(A);
        delete_matrix(B);
        delete_matrix(C);
    }
}
```

- [ ] **Step 3: 注册测试套件到 `tests/test_main.cpp`**

```cpp
#include "test_framework.h"

#include <iostream>

void test_matrix_utils_suite();
void test_timer_suite();
void test_transpose_suite();

int main() {
    test_matrix_utils_suite();
    test_timer_suite();
    test_transpose_suite();

    std::cout << "\n" << g_tests_run << " checks run, " << g_tests_failed << " failed.\n";
    return g_tests_failed == 0 ? 0 : 1;
}
```

- [ ] **Step 4: 更新 `CMakeLists.txt`（库新增 `src/transpose.cpp`，测试新增 `tests/test_transpose.cpp`）**

```cmake
add_library(matrix_op_lib
    src/matrix_utils.cpp
    src/transpose.cpp
)

# ...

add_executable(matrix_tests
    tests/test_main.cpp
    tests/test_matrix_utils.cpp
    tests/test_timer.cpp
    tests/test_transpose.cpp
)
```

- [ ] **Step 5: 构建，确认因缺少实现而链接失败**

Run:
```powershell
cmake -S . -B build
cmake --build build --config Release
```
Expected: 链接错误，`transpose_baseline` / `transpose_optimized` 未解析。

- [ ] **Step 6: 编写 `src/transpose.cpp`**

```cpp
#include "transpose.h"

#include <algorithm>

void transpose_baseline(const float* A, float* B, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

void transpose_optimized(const float* A, float* B, int rows, int cols, int BLOCK_SIZE) {
    for (int ii = 0; ii < rows; ii += BLOCK_SIZE) {
        int iMax = std::min(ii + BLOCK_SIZE, rows);
        for (int jj = 0; jj < cols; jj += BLOCK_SIZE) {
            int jMax = std::min(jj + BLOCK_SIZE, cols);
            for (int i = ii; i < iMax; ++i) {
                for (int j = jj; j < jMax; ++j) {
                    B[j * rows + i] = A[i * cols + j];
                }
            }
        }
    }
}
```

- [ ] **Step 7: 重新构建，确认测试通过**

Run:
```powershell
cmake --build build --config Release
.\build\Release\matrix_tests.exe
```
Expected: `N checks run, 0 failed.`，退出码 0。

- [ ] **Step 8: Commit**

```bash
git add include/transpose.h src/transpose.cpp tests/test_transpose.cpp tests/test_main.cpp CMakeLists.txt
git commit -m "Implement cache-blocked matrix transpose"
```

---

### Task 5: multiply — 基准与分块优化实现（TDD）

**Files:**
- Create: `include/multiply.h`
- Create: `src/multiply.cpp`
- Create: `tests/test_multiply.cpp`
- Modify: `tests/test_main.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `create_matrix`, `delete_matrix`, `fill_matrix`, `compare_matrices`（Task 2）。
- Produces:
  ```cpp
  void multiply_baseline(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
  void multiply_optimized(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB, int BLOCK_SIZE);
  ```
  Task 7 的 `main.cpp` 直接调用这两个函数。

- [ ] **Step 1: 编写 `include/multiply.h`**

```cpp
#pragma once

void multiply_baseline(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
void multiply_optimized(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB, int BLOCK_SIZE);
```

- [ ] **Step 2: 编写失败测试 `tests/test_multiply.cpp`**

```cpp
#include "test_framework.h"
#include "multiply.h"
#include "matrix_utils.h"

void test_multiply_suite() {
    // multiply_baseline 对已知小矩阵计算正确
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

    // BLOCK_SIZE 能整除维度时，multiply_optimized 与 baseline 在方阵上结果一致
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

    // 非方阵、BLOCK_SIZE 不能整除任一维度时（触发边界裁剪），结果依然一致
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
```

- [ ] **Step 3: 注册测试套件到 `tests/test_main.cpp`**

```cpp
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
```

- [ ] **Step 4: 更新 `CMakeLists.txt`（库新增 `src/multiply.cpp`，测试新增 `tests/test_multiply.cpp`）**

```cmake
add_library(matrix_op_lib
    src/matrix_utils.cpp
    src/transpose.cpp
    src/multiply.cpp
)

# ...

add_executable(matrix_tests
    tests/test_main.cpp
    tests/test_matrix_utils.cpp
    tests/test_timer.cpp
    tests/test_transpose.cpp
    tests/test_multiply.cpp
)
```

- [ ] **Step 5: 构建，确认因缺少实现而链接失败**

Run:
```powershell
cmake -S . -B build
cmake --build build --config Release
```
Expected: 链接错误，`multiply_baseline` / `multiply_optimized` 未解析。

- [ ] **Step 6: 编写 `src/multiply.cpp`**

```cpp
#include "multiply.h"

#include <algorithm>
#include <cstring>

void multiply_baseline(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
}

void multiply_optimized(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB, int BLOCK_SIZE) {
    std::memset(C, 0, static_cast<size_t>(rowsA) * colsB * sizeof(float));

    for (int ii = 0; ii < rowsA; ii += BLOCK_SIZE) {
        int iMax = std::min(ii + BLOCK_SIZE, rowsA);
        for (int kk = 0; kk < colsA; kk += BLOCK_SIZE) {
            int kMax = std::min(kk + BLOCK_SIZE, colsA);
            for (int jj = 0; jj < colsB; jj += BLOCK_SIZE) {
                int jMax = std::min(jj + BLOCK_SIZE, colsB);
                for (int i = ii; i < iMax; ++i) {
                    for (int k = kk; k < kMax; ++k) {
                        float a = A[i * colsA + k];
                        for (int j = jj; j < jMax; ++j) {
                            C[i * colsB + j] += a * B[k * colsB + j];
                        }
                    }
                }
            }
        }
    }
}
```

`multiply_optimized` 采用分块 + 块内 `ikj` 循环序（而非朴素 `ijk`），使块内对 `B`、`C` 的访问按行连续，是行主序存储下缓存友好的标准写法。

- [ ] **Step 7: 重新构建，确认测试通过**

Run:
```powershell
cmake --build build --config Release
.\build\Release\matrix_tests.exe
```
Expected: `N checks run, 0 failed.`，退出码 0。

- [ ] **Step 8: Commit**

```bash
git add include/multiply.h src/multiply.cpp tests/test_multiply.cpp tests/test_main.cpp CMakeLists.txt
git commit -m "Implement cache-blocked matrix multiply with ikj loop order"
```

---

### Task 6: main.cpp — 转置实验驱动

**Files:**
- Modify: `src/main.cpp`

**Interfaces:**
- Consumes: `create_matrix`, `clone_matrix`, `delete_matrix`, `fill_matrix`, `compare_matrices`（Task 2）；`run_benchmark`（Task 3）；`transpose_baseline`, `transpose_optimized`（Task 4）。
- Produces: 无新接口，本任务只产出可执行的实验输出。

- [ ] **Step 1: 编写 `src/main.cpp`（仅转置实验，乘法实验留待 Task 7）**

```cpp
#include "matrix_utils.h"
#include "timer.h"
#include "transpose.h"

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
        float* B = clone_matrix(A, cols, rows);
        float* C = clone_matrix(A, cols, rows);

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

}  // namespace

int main() {
    run_transpose_experiment();
    return 0;
}
```

- [ ] **Step 2: 构建**

Run:
```powershell
cmake --build build --config Release
```
Expected: 编译链接成功。

- [ ] **Step 3: 运行并人工核验输出**

Run:
```powershell
.\build\Release\matrix_bench.exe
```
Expected: 打印 `=== Experiment 1: Matrix Transpose ===` 表格，5 个规模 × 4 个 BLOCK_SIZE 共 20 行数据，`Correct` 列全部为 `YES`，且总体上 `Optimized(ms)` 应小于 `Baseline(ms)`（规模越大分块优势越明显；1000x4100 那一行验证了边界裁剪路径）。

- [ ] **Step 4: Commit**

```bash
git add src/main.cpp
git commit -m "Wire up transpose experiment driver in main.cpp"
```

---

### Task 7: main.cpp — 乘法实验驱动 + 完整程序验证

**Files:**
- Modify: `src/main.cpp`

**Interfaces:**
- Consumes: `create_matrix`, `delete_matrix`, `fill_matrix`, `compare_matrices`（Task 2）；`run_benchmark`（Task 3）；`multiply_baseline`, `multiply_optimized`（Task 5）；沿用 Task 6 中定义的 `kBlockSizes`, `kRepetitions`, `print_table_header`。
- Produces: 无新接口，完成完整的双实验驱动程序。

- [ ] **Step 1: 修改 `src/main.cpp`，新增乘法实验**

在 `#include "transpose.h"` 之后新增 `#include "multiply.h"`；在匿名命名空间内 `run_transpose_experiment` 函数之后新增：

```cpp
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
```

并将 `main` 函数改为：

```cpp
int main() {
    run_transpose_experiment();
    run_multiply_experiment();
    return 0;
}
```

- [ ] **Step 2: 更新 `CMakeLists.txt`，把 `multiply.h` 的依赖接入 `matrix_bench`**

`matrix_bench` 已经通过 `target_link_libraries(matrix_bench PRIVATE matrix_op_lib)` 链接了 `matrix_op_lib`（其中已包含 `src/multiply.cpp`，见 Task 5），且 `include_directories(include)` 已经让 `multiply.h` 可见，因此 **本步骤无需修改 `CMakeLists.txt`**——只需确认这一点，不做改动。

- [ ] **Step 3: 构建**

Run:
```powershell
cmake --build build --config Release
```
Expected: 编译链接成功。

- [ ] **Step 4: 运行完整程序并核验输出**

Run:
```powershell
.\build\Release\matrix_bench.exe
```
Expected:
- 先打印转置实验表格（同 Task 6），共 20 行，`Correct` 全 `YES`。
- 再打印 `=== Experiment 2: Matrix Multiply ===` 表格，5 个规模 × 4 个 BLOCK_SIZE 共 20 行，`Correct` 全 `YES`。
- 1024×1024×1024 那组因为是 O(n³) 且 baseline 未做任何缓存优化，预计单次运行耗时可能达到数百毫秒到数秒；5 组规模 × (1 baseline + 4 optimized) × 6 次调用（1 预热 + 5 计时）累计运行整个程序可能需要几分钟，属于预期行为（这是一次性运行的基准程序，不是需要频繁执行的单元测试）。
- 500×1000×250 那一行验证了乘法分块的边界裁剪路径。
- 整体应观察到 `Speedup(x)` 大多 > 1（optimized 更快），规模越大越明显；如果 BLOCK_SIZE 相对 L1/L2 缓存过大或过小，个别行速度提升可能不明显，这是真实的性能特征，不代表实现有误。

- [ ] **Step 5: 完整跑一次单元测试套件做最终回归确认**

Run:
```powershell
.\build\Release\matrix_tests.exe
```
Expected: `N checks run, 0 failed.`，退出码 0。

- [ ] **Step 6: Commit**

```bash
git add src/main.cpp
git commit -m "Wire up matrix multiply experiment driver in main.cpp"
```

---

## Self-Review Notes

- **Spec coverage**：设计文档中的五个公共接口（Task 2）、转置基准/优化实现（Task 4）、乘法基准/优化实现（Task 5）、两个实验的测试流程与规模表（Task 6、Task 7）、`-O3` 到 MSVC 的映射（Task 1 CMakeLists）、容差比较（Task 2）、计时框架（Task 3）均有对应任务覆盖。
- **占位符检查**：全文无 TBD/TODO，所有步骤均给出完整代码。
- **类型一致性**：`transpose_baseline/transpose_optimized`、`multiply_baseline/multiply_optimized`、`run_benchmark`、`compare_matrices` 等函数签名在 Task 2/3/4/5 中定义后，Task 6/7 的 `main.cpp` 调用与之完全一致；`fill_matrix` 的 `static` 局部 RNG 设计在 Task 2 中明确记录了偏离原设计文档描述之处（design 文档写的是"每次调用重新播种为同一固定种子"，实现改为"只播种一次、持续推进"，原因是前者会让形状相同的相邻两次 `fill_matrix` 调用产出完全相同的数据，削弱乘法正确性测试的有效性；效果仍然是确定性可复现的，因为调用顺序在程序中是固定的）。
