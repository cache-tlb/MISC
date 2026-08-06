# Alpha Mask Mipmap 修正 Demo

用 WebGL2 对比 alpha mask 纹理在三种 mipmap 策略下的表现：**无 mipmap** / **默认 box filter mipmap** / **覆盖率修正的 mipmap**。

修正方法来自 Ignacio Castaño，[Computing Alpha Mipmaps](https://www.ludicon.com/castano/blog/articles/computing-alpha-mipmaps/)。

## 运行

需要通过 HTTP 打开（`fetch` 在 `file://` 下会被 CORS 拦掉）：

```bash
python -m http.server 8113
```

然后访问 `http://localhost:8113/`。

## 文件

| 文件 | 作用 |
|---|---|
| `src/alpha-mipmap.js` | **核心**：mip 链生成 + 覆盖率修正 |
| `src/renderer.js` | WebGL2 渲染，shader、场景、纹理上传 |
| `src/obj-loader.js` | 自写的 OBJ / MTL 解析器 |
| `src/camera.js` | UE 风格飞行相机 |
| `src/main.js` | 资源加载、UI 绑定、渲染循环 |
| `src/m4.js` | 矩阵库，未改动 |
| `src/webgl-utils.js` | WebGL 工具函数，未改动 |

---

## 问题

alpha 测试的写法是 `if (a < Ar) discard;`。决定画面观感的不是平均 alpha，而是**覆盖率**——通过测试的纹素占比：

```
coverage = count(a_i >= Ar) / N
```

box filter 求的是平均值，而平均会把值往中间拉。对树叶这种「大部分纹素非 0 即 1」的 mask，每降一级 mipmap 覆盖率都会偏离原值，于是树冠随距离变远而变形，最后整片消失。

本 demo 用的 `DB2X2_L01.png`（1024×1024）alpha 分布：**68.4%** 全透明、**24.2%** 全不透明，中间过渡只占 7.4%。典型的树叶 mask。

## 修正方法

不直接搜索 scale（它无界），而是搜索**这一级 mip 若想维持原覆盖率所需的参考值 `ar`**：

```
count(a_i >= ar) / N = coverage(level 0)
```

`ar` 落在 `[0, 1]` 内，可以二分。解出后换算成 scale：

```
scale = Ar / ar
```

把这一级的 alpha 整体乘以 `scale`，它在真实阈值 `Ar` 下的覆盖率就回到了 level 0 的水平。

## 代码怎么组织的

`alpha-mipmap.js` 把工作拆成**与阈值无关**和**与阈值相关**两部分，因为 demo 里阈值是可以拖滑块实时改的。

### 一次性预处理 — `prepare()`

```
extractChannel()   PNG 的 A 通道 → Float32Array，值域 [0,1]
       ↓
buildChain()       逐级 2×2 box filter，一直降到 1×1（1024 → 11 级）
       ↓
buildCoverageCdf() 每级统计一份 256 bin 的 alpha 直方图，再求后缀和
```

两个实现上的选择：

**① 覆盖率按 GPU 真正的采样方式估计，不是简单数纹素。**
GPU 是双线性采样的，一个落在「不透明纹素」和「透明纹素」之间的 tap 仍可能通过测试。所以对每个 2×2 纹素小块做 `SUBSAMPLES × SUBSAMPLES`（默认 2×2）次双线性插值采样再统计。这与 NVTT 的做法一致。

**② 直方图是为了让阈值可交互。**
把超采样结果累积成直方图并求后缀和后，`cdf[b]` 就是「量化到 8bit 后 ≥ b 的采样占比」。于是任意阈值下的覆盖率退化成一次查表 + 线性插值（`coverageAt()`），二分求解不再需要重新扫描像素。

代价对比（1024×1024，实测）：`prepare()` 约 **80 ms**（只在加载时跑一次），之后每次改阈值重解整条链只要**几微秒**。所以滑块的 `input` 事件可以直接触发重算，不需要防抖。

**③ mip 链始终从「未缩放」的父级生成。**
`buildChain()` 生成的是干净的 box filter 链，缩放只在最后写入纹理时施加（`toUint8(level, scale)`）。如果拿缩放后的结果继续降采样，各级的修正量会互相累积放大。

### 每次阈值变化 — `solveChain()`

```js
targetCoverage = coverageAt(cdfs[0], alphaRef);   // level 0 的覆盖率就是目标

for each level > 0:
    // 二分 ar。覆盖率对 ar 单调不增：通过的太多 → ar 要调大
    lo = 0, hi = 1, ar = alphaRef
    repeat BISECTION_STEPS (24) times:
        c = coverageAt(cdf, ar)
        if (c > target) lo = ar
        else if (c < target) hi = ar
        else break
        ar = (lo + hi) / 2

    scale = clamp(alphaRef / ar, 1/16, 16)
```

clamp 是必要的：如果某一级的覆盖率无论如何都达不到目标，二分会把 `ar` 推向 0，`scale` 就会跑向无穷。

`main.js` 拿到 scale 后重建纹理数据并原地更新：

```js
// toUploadable() 内部就是 alphaMipmap.toUint8(level, scale)：
// 乘上 scale，饱和截断，量化回 8bit
correctedLevels = mask.levels.map((level, i) =>
    toUploadable(level, solutions[i].scale));
renderer.updateMaskTexture(maskTextures[MODE_CORRECTED], correctedLevels);
```

### 三种模式怎么接的

三张纹理都是 **R8 单通道**，逐级 `texImage2D` 显式上传，只有 mask 在切换，albedo（RGB8）和其他一切都不变：

| 模式 | 内容 | MIN_FILTER |
|---|---|---|
| 无 mipmap | 只上传 level 0 | `LINEAR` |
| 默认 mipmap | 完整 box filter 链，scale 全为 1 | `LINEAR_MIPMAP_LINEAR` |
| 修正的 mipmap | 同一条链，逐级乘上解出的 scale | `LINEAR_MIPMAP_LINEAR` |

shader 里就是一句：

```glsl
if (u_useMask > 0.5 && mask < u_alphaRef) discard;
```

## 效果

阈值取默认的 0.33，实测逐级覆盖率：

| Level | 尺寸 | 默认 mipmap | scale | 修正后 |
|---|---|---|---|---|
| 0 | 1024² | 28.5% | 1.000 | 28.5% |
| 1 | 512² | 29.1% | 0.889 | 28.5% |
| 2 | 256² | 30.2% | 0.816 | 28.5% |
| 3 | 128² | 32.8% | 0.750 | 28.4% |
| 4 | 64² | 37.3% | 0.705 | 28.4% |
| 5 | 32² | 40.8% | 0.735 | 28.3% |
| 6 | 16² | 39.6% | 0.867 | 27.6% |
| 7 | 8² | 24.0% | 1.020 | 28.1% |
| 8 | 4² | **0.0%** | 1.132 | 25.0% |
| 9 | 2² | **0.0%** | 1.188 | 0.0% |
| 10 | 1² | **0.0%** | 1.190 | 0.0% |

**阈值决定默认 mipmap 往哪个方向跑偏**，这一点值得注意：

- 阈值 **< 0.5**：box filter 让 mask 逐级「膨胀」（上表 28.5% → 40.8%），叶片先变胖糊成一团，到 4×4 那级突然塌成 0。
- 阈值 **> 0.5**：变成逐级「腐蚀」。0.66 时覆盖率单调下降，16×16 那级就已经归零，树冠很早就整片消失。

两个方向修正后都保持在目标值附近。面板里的「逐级覆盖率 / 缩放系数」表格会实时显示这些数字。

### 方法的固有上限

上表最粗的三级值得说明。当一级 mip 已经降采样成接近常数时，它的覆盖率只可能是 0% 或 100%，再怎么缩放 alpha 也回不到 28.5%——这不是实现 bug，是该方法本身的边界。

代码对此是**如实报告**的：表格里的覆盖率用 `coverageAfterScale()` 计算，它按「乘 scale 再量化成 8bit」的真实结果统计，所以 2×2 和 1×1 会老实显示 0%，而不是假装被修好了。8×8 及以上的级别，最坏残差在 3.4% 以内。

---

## 其他两个非显然的实现点

读代码时容易困惑的地方，一并记在这里。

**坐标系是右手系**，与 OpenGL 常规一致：x 右、y 上、**z 指向屏幕外朝向观察者**，所以相机看向 **-z**。用的就是 `m4.js` 原有的 `lookAt` / `perspective`，没有改动这个文件。

相机的 yaw 是绕 +y 按右手定则的旋转，因此 **yaw 增大是向左转**，yaw = 0 时正对 -z。鼠标右移要向右转，所以 `camera.yaw -= movementX * sensitivity`。三个视角预设都放在树的 +z 侧，平行光方向也偏 +z，保证你正对着的那一面是被照亮的。

**树叶的法线不用 `gl_FrontFacing` 判断**，而是直接对视线方向定向：

```glsl
if (u_twoSided > 0.5 && dot(normal, u_cameraPosition - v_worldPosition) < 0.0) {
  normal = -normal;
}
```

原因不是手系：`gl_FrontFacing` 回答的是「你在 card 的哪一面」，只有当顶点法线垂直于 card 时它才等价于「法线在哪个半球」。这棵树的叶片法线是**刻意偏离 card 平面**的（为了让树冠光照更柔和），两者在约 15% 的叶片上给出不同答案。实测五个机位，用视线判据的正确率是 **100%**，用 `gl_FrontFacing` 是 83~85%。

只对树叶（单面 card）生效。树干是闭合实体，法线本来就对，在它的轮廓边缘做翻转反而会产生光照接缝。

## 交互

按住**鼠标右键**激活飞行（与 UE 视口一致）：`W`/`S` 前后、`A`/`D` 平移、`E`/`Q` 升降、`Shift` 加速、滚轮调速，鼠标控制 yaw / pitch。

快捷键：`1`/`2`/`3` 切换 mipmap 策略，`C` 开关并排对比，`H` 收起面板。
