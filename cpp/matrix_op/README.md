# matrix_op

C++ 性能对比实验框架：对比矩阵转置、矩阵乘法的朴素实现与缓存分块（cache blocking）优化实现，测量并输出两者的耗时、加速比与正确性。

## 项目做的事情

矩阵统一用 `float` 类型、行主序（row-major）存储，`A[i * cols + j]` 访问第 i 行第 j 列元素。

提供的公共接口（`include/matrix_utils.h` / `src/matrix_utils.cpp`）：

- `create_matrix(rows, cols)` — 创建全零矩阵
- `clone_matrix(matrix, rows, cols)` — 克隆矩阵
- `delete_matrix(matrix)` — 释放矩阵
- `fill_matrix(matrix, rows, cols)` — 用 `[-1, 1]` 随机浮点数填充
- `compare_matrices(A, B, rows, cols)` — 带容差的近似比较（因为分块乘法会改变浮点累加顺序）

两组对比实验：

1. **矩阵转置**（`include/transpose.h`）：`transpose_baseline` 双重循环 vs. `transpose_optimized` 按 `BLOCK_SIZE` 分块。
2. **矩阵乘法**（`include/multiply.h`）：`multiply_baseline` 朴素三重循环 vs. `multiply_optimized` 分块 + 块内 `ikj` 循环序。

计时统一通过 `include/timer.h` 的 `run_benchmark(func, repetitions=5, warmup=true)`：1 次预热（不计时）+ 5 次正式计时取平均值。

`src/main.cpp` 驱动两组实验，各自在 5 组矩阵规模 × `BLOCK_SIZE ∈ {16, 32, 64, 128}` 下运行，打印结果表（耗时、加速比、正确性）：

- 转置规模：512², 1024², 2048², 4096², 1000×4100（非方阵，验证边界裁剪）
- 乘法规模：128², 256², 512², 1024², 500×1000×250（非方阵，验证边界裁剪）

设计文档与实现计划见 [`docs/superpowers/specs/`](docs/superpowers/specs/) 和 [`docs/superpowers/plans/`](docs/superpowers/plans/)。

## 编译

依赖：CMake ≥ 3.15，一个 C++17 编译器（本机用的是 MSVC / Visual Studio 2022；若有 g++/clang++ 会自动使用 `-O3 -march=native`）。

```powershell
cmake -S . -B build
cmake --build build --config Release
```

第一条命令会自动检测生成器和编译器（Windows 上通常是 "Visual Studio 17 2022"），不需要手动进入 vcvars 环境。修改代码后重新编译，只需重复第二条命令；只有改动了 `CMakeLists.txt` 才需要重新执行第一条。

构建产物：

- `build/Release/matrix_bench.exe` — 基准测试可执行文件
- `build/Release/matrix_tests.exe` — 单元测试可执行文件

## 运行

```powershell
.\build\Release\matrix_bench.exe
```

依次输出转置实验和矩阵乘法实验的结果表。乘法实验包含 1024×1024×1024 的未优化三重循环基准，整体运行可能需要几分钟，属正常现象。

```powershell
.\build\Release\matrix_tests.exe
```

运行单元测试（覆盖 matrix_utils、timer、transpose、multiply），预期输出 `N checks run, 0 failed.`，退出码 0。

## 项目结构

```
matrix_op/
├── CMakeLists.txt
├── include/
│   ├── matrix_utils.h    # 公共矩阵接口
│   ├── timer.h            # run_benchmark 计时工具
│   ├── transpose.h        # 转置基准 / 优化实现
│   └── multiply.h         # 乘法基准 / 优化实现
├── src/
│   ├── matrix_utils.cpp
│   ├── transpose.cpp
│   ├── multiply.cpp
│   └── main.cpp            # 实验驱动，打印结果表
└── tests/                   # 自定义轻量断言框架 + 各模块单元测试
```
