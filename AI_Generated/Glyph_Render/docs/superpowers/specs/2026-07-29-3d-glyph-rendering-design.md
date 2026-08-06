# 设计文档：Scanline Sweeper 的 3D 字形渲染支持

- 日期：2026-07-29
- 前置设计：`docs/superpowers/specs/2026-07-23-scanline-sweeper-vs-sdf-demo-design.md`
- 参考论文：`reference/scanline_sweeper_preprint.pdf`（§3.2 逐像素窗口、§4.4 Accurate Footprint Assembly）
- 文本抽取：`reference/out.txt`

## 1. 目标

让 demo 支持**字形附着在 3D 空间平面上**的渲染：文字躺在一块三维平面上，相机可轨道旋转、推拉，透视投影。分屏擦除对比（左 Sweeper / 右 SDF-MSDF）保留，两侧共用同一个 MVP，跨分界线仍逐像素对齐。

前置设计的非目标之一是"不做透视/剪切变换（论文 §4.4 扩展）；仅 2D 正交"。本设计**解除**这一条。

## 2. 核心问题：足迹从哪来

`scanlineSweep()` 的契约是「给定一个 em 空间的**轴对齐矩形窗口**，返回无权重盒式滤波覆盖率」。这个契约与投影方式无关，因此**核心 sweep 数学一行都不用改**。

真正被写死成 2D 正交的是**窗口从哪来**。现状：

```glsl
uniform float u_emPerPixel;                       // 全屏一个标量
vec2 size = vec2(u_emPerPixel);                   // 各向同性方窗
vec2 offset = v_localEm - 0.5*size;
```

论文 §3.2 说得很直白：正交投影下这个窗口在 draw 时就已知；一般情况要用 `fwidth` 逐像素估计；**"the footprint is dynamically varying per-pixel, as this is what permits anti-aliased glyph shading under arbitrary transformations."**

3D 下，一个像素在 em 空间的足迹是一个**平行四边形**

$$P=\{\,c+u\cdot\mathbf{d}_x+v\cdot\mathbf{d}_y \;:\; u,v\in[-\tfrac12,\tfrac12]\,\},\qquad \mathbf{d}_x=\mathrm{dFdx}(\texttt{v\_localEm}),\ \ \mathbf{d}_y=\mathrm{dFdy}(\texttt{v\_localEm})$$

因为 `v_localEm` 是 em 空间坐标、由硬件做**透视校正插值**，这两个导数天然包含了透视压缩、旋转与非均匀缩放——不需要手写雅可比，也不需要知道相机参数。

## 3. 两级足迹装配（论文 §4.4）

| 级别 | 窗口 | 说明 |
|---|---|---|
| **L1** 轴对齐矩形 | `size = fwidth(v_localEm)` = `abs(ddx)+abs(ddy)` | 即 $P$ 的 AABB。论文推荐的最简方案，"fairly passable" |
| **L2** 各向异性多窗口 | 沿长轴切 $N$ 片，每片取自身 AABB，逐窗口 sweep 后平均 | 论文 §4.4 的椭圆分解思路，落到轴对齐矩形窗口上 |

**L1 为什么在掠射角失效**：AABB 是 $P$ 的外接盒。正入射时 $\mathbf{d}_x=(e,0)$、$\mathbf{d}_y=(0,-e)$，AABB 面积 $e^2$ 与 $|P|=e^2$ 相等，无损；但当 $P$ 被透视压成一个极扁的斜平行四边形时，AABB 面积可以是 $|P|$ 的数倍，多出来的部分全是不该被采进来的墨——表现为发糊。

**L2 分解规则**（`footprintWindows`）：

1. 取 $\mathbf{d}_x,\mathbf{d}_y$ 中较长者为 $\mathbf{M}$（长轴），另一者为 $\mathbf{m}$（短轴）；
2. 各向异性比 $\rho=|\mathbf{M}|/|\mathbf{m}|$，窗口数 $N=\mathrm{clamp}(\lceil\rho\rceil,\,1,\,N_{\max})$；
3. 第 $i$ 个子平行四边形（$i=0..N-1$）：
   $$c_i=c+\mathbf{M}\Bigl(\frac{i+0.5}{N}-\frac12\Bigr),\qquad P_i=\Bigl\{c_i+u\cdot\frac{\mathbf{M}}{N}+v\cdot\mathbf{m}\Bigr\}$$
   这 $N$ 片**无缝无重叠地精确铺满** $P$；
4. 第 $i$ 个窗口取 $P_i$ 的 AABB：
   $$\texttt{size}=\frac{|\mathbf{M}|_{\text{分量}}}{N}+|\mathbf{m}|_{\text{分量}},\qquad \texttt{offset}_i=c_i-\tfrac12\,\texttt{size}$$
   $N$ 个窗口**尺寸相同**，只有偏移不同。

**关键性质**：$N=1$ 时 `size` $=|\mathbf{d}_x|+|\mathbf{d}_y|=$ `fwidth`，**精确退化为 L1**。因此 2D 正交下 $\rho=1\Rightarrow N=1$，多窗口路径**零开销、零行为差异**，无需为 2D/3D 写两条 shader 路径。

**覆盖率合成**：

$$\text{coverage}=\frac1N\sum_{i=0}^{N-1}\mathrm{sat}\Bigl(\frac{\text{sign}\cdot\sum_C \text{sweep}(C,\ \texttt{size},\ \texttt{offset}_i)}{\texttt{size}_x\cdot\texttt{size}_y}\Bigr)$$

**先逐窗口 clamp 再平均**，而非先求和后 clamp——保留现有的逐窗口 overlap clamp 语义（前置设计 §11），避免自交轮廓在某个子窗口溢出后污染整体均值。

**成本**：$N\times$ 曲线循环。$N_{\max}$ 由 UI 控制（1–8，默认 4）。

## 4. 变换：三个 uniform 合并为一个 `u_mvp`

两个 renderer 的 `u_pxPerEm` / `u_originDev` / `u_backing` 一并替换为单个 `uniform mat4 u_mvp`：

```glsl
vec2 worldEm = localEm + a_emOrigin;
gl_Position = u_mvp * vec4(worldEm, 0.0, 1.0);   // 字形躺在平面局部坐标的 z=0
```

**2D 模式必须像素级不变**。由现有正交参数直接构造等价矩阵（列主序）：

```
clip.x = ex·(2·pxPerEm/bw) + (2·ox/bw − 1)
clip.y = ey·(2·pxPerEm/bh) + (1 − 2·oy/bh)
```

即 `mat4` 的 `m00 = 2·pxPerEm/bw`、`m11 = 2·pxPerEm/bh`、`m30 = 2·ox/bw − 1`、`m31 = 1 − 2·oy/bh`，其余为单位阵。y 翻转被 `m11` 的符号与 `m31` 一起吸收（原式对 em 的 y 取负、又对 device y 取负，两次翻转抵消为正号）。

**3D 模式**：`u_mvp = perspective(fov, aspect, near, far) × orbitView(yaw, pitch, dist) × planeModel`。

- `planeModel`：把文本块在 em 空间的包围中心平移到原点。**em 与 world 取 1:1**（`worldPerEm = 1`），文字躺在平面局部的 z=0、以原点为中心。
- `orbitView`：`eye = dist·(sin(yaw)cos(pitch), sin(pitch), cos(yaw)cos(pitch))`，`target = 原点`，`up = (0,1,0)`。`yaw=pitch=0` → 相机在 +Z 正对平面。pitch 夹到 ±85° 以避开 up 向量退化，`dist` 夹到 [0.3, 20]。
- **`dist` 是取景距离的倍数，不是世界单位**（实现期修正，见下）。`fitDistance(bounds, fov, aspect)` 给出恰好框住文本块的距离，实际相机距离 = `dist × fitDistance`。
- 预设机位：**正面** `(yaw 0°, pitch 0°, dist 1.15)`、**45°** `(45°, 20°, 1.30)`、**极端掠射** `(80°, 12°, 1.35)`。实测屏幕中心各向异性比：正面 ρ=1.00，掠射 ρ=5.75，落在目标区间 5–10。
- **字号滑杆在 3D 下通过相机生效**：`有效 dist = dist × (120 / fontSizePx)`。字号越大相机越近，屏幕上每 em 覆盖的像素越多——与 2D 下"字号提高像素密度"的语义一致，也让 Sweeper/SDF 的对比在 3D 下依然可调。`zoom` 在 3D 下不参与。

> **实现期修正**：本节原定 `worldPerEm = fontSizePx/200` 且预设 `dist` 为绝对世界单位。该方案不成立——默认三行文本约 10 em 宽，`fontSizePx=120` 时即 6 世界单位，而 `dist=2.6`、`fov=50°` 的相机只能看到约 4.3 单位宽，文字必然溢出画面；且换一段文本或改一次字号，取景就得重调。改为"取景倍数 + 字号通过相机生效"后，预设对任意文本长度与字号都成立，由 `test/scene3d.test.mjs` 的取景测试守住（宽块/窄块/方块三种极端 extent 的四个角都必须落在视锥内）。
- 深度测试全程关闭（共面、无自遮挡），不做背面剔除（背面可见为镜像文字，符合预期）。绘制顺序：网格平面 → Sweeper（左半）→ SDF（右半）。

## 5. SDF/MSDF 侧同步改造

SDF 的抗锯齿宽度同样来自那个标量 uniform：`aa = 0.5*u_emPerPixel`。改为逐像素：

```glsl
aa = 0.5 * max(fwidth(v_localEm).x, fwidth(v_localEm).y);
```

2D 下 `fwidth(v_localEm) = (emPerPixel, emPerPixel)`，`max` 即 `emPerPixel`，**与现状数值完全等价**。SDF 的 VS 需新增 `out vec2 v_localEm`（现在只输出 `v_uv`）。

SDF 侧**不做**多窗口：它每个 texel 只有一个各向同性的距离值，没有可供各向异性积分的信息。这个不对称本身就是对比的看点——掠射角下 Sweeper 能靠多窗口收紧足迹，SDF 只能糊。

## 6. 文件结构

新增：

```
js/mat4.js                 最小 mat4：identity/mul/perspective/lookAt/ortho2d，纯函数
js/scene3d.js              轨道相机状态 → MVP；机位预设；纯函数
js/sweeper-footprint.js    足迹装配层：footprintWindows / coverageFootprint（GLSL 镜像）
js/plane-renderer.js       3D 背景网格平面（~40 行，procedural grid + fwidth 线宽 AA）
test/mat4.test.mjs
test/sweeper-footprint.test.mjs
docs/blog/figs/fig5-footprint-3d.svg
```

修改：

```
js/sweeper-renderer.js   u_mvp；FS 内联 footprintWindows 的多窗口外层循环；u_maxWindows
js/sdf-renderer.js       u_mvp；v_localEm；逐像素 aa
js/main.js               2D/3D 模式、轨道输入、预设、aniso 滑杆、deep link、info 读出
index.html               新控件
test/_helpers.mjs        上移 sweeper-core.test.mjs 里的 windingInside/supersampledCoverage
test/sweeper-core.test.mjs  改为 import 上移后的 helper
README.md                控制项/结构/deep link/局限/测试数
docs/blog/scanline-sweeper.md  新增 3D 一节 + 局限补充
```

**`js/sweeper-core.js` 不修改**。分层刻意与论文一致：core 只负责「一个矩形窗口的覆盖率」，足迹怎么拼是上层的事（§4.4：*"does not stipulate any particular footprint assembly"*）。`sweeper-footprint.js` 依赖 core 的 `sumSweep`，反向无依赖。

## 7. 数据流与重建契约（扩展前置设计 §6）

| 变化项 | 重建内容 |
|---|---|
| 文本 / 字体 | 同前：重新布局 + 重建曲线纹理 + 图集 + instance 缓冲 |
| SDF 分辨率 / 模式 | 同前 |
| **2D/3D 模式、yaw/pitch/dist/fov、$N_{\max}$** | **仅改 uniform**，不重建任何缓冲/纹理 |
| 平移 / 缩放 / 字号 / gamma | 同前，仅改 uniform |

3D 相机交互全部落在 uniform 上，前置设计的重建契约不被破坏。

`view` 对象契约随之变化：`{ pxPerEm, originDev, backing, emPerPixel, gamma, color }` → `{ mvp: Float32Array(16), gamma, color, maxWindows }`。

## 8. UI（`index.html` + `main.js`）

- **视图模式**分段按钮 `2D | 3D`，默认 **2D**（不破坏现有 deep link、README、截图）。
- 3D 专属控件（切到 3D 才显示）：机位预设三按钮、FOV 滑杆。
- **各向异性窗口上限 $N_{\max}$** 滑杆 1–8（默认 4），2D/3D 都显示——2D 下 $\rho=1$ 使其不产生开销，这本身可演示。
- 交互语义按模式分派：2D 拖拽=平移、滚轮=缩放（现状不变）；3D 拖拽=轨道旋转、滚轮=推拉。
- **重置视图**按当前模式重置。
- info 读出增加：当前模式、$N_{\max}$、屏幕中心处的各向异性比 $\rho$（由 CPU 侧同一套 `scene3d` 数学估算，便于把"为什么要多窗口"量化）。
- deep link 新增 `view=3d`、`yaw`、`pitch`、`dist`、`aniso`。

## 9. 验证计划

`test/mat4.test.mjs`：

1. `mul` 的单位元与结合律；`perspective` 把已知点映到已知 clip 坐标。
2. **2D 等价性**：随机 em 点 × 随机 `(pxPerEm, originDev, backing)`，`ortho2d` 矩阵变换结果与旧内联公式误差 < 1e-6。这条守住"2D 像素级不变"。

`test/sweeper-footprint.test.mjs`：

3. **各向同性退化**：$\mathbf{d}_x=(e,0),\mathbf{d}_y=(0,-e)$ 时 $N=1$，且 `coverageFootprint` 与现有 `coverage` 结果完全一致。
4. **铺满性**：$N$ 个子平行四边形面积和 $=|\mathbf{d}_x\times\mathbf{d}_y|$；中心沿长轴等距；各窗口尺寸相同；$N\le N_{\max}$。
5. **精度提升（核心验收）**：真实字形（Tinos `'e'`）+ 强各向异性足迹（$\rho\approx 8$），以**对真实平行四边形做分层超采样 + 非零绕数**为 ground truth：
   - `MAE(多窗口) < MAE(单 AABB 窗口)`，且有明确余量；
   - `MAE(多窗口)` 低于绝对阈值。

   这条直接验证"L2 确实解决了掠射角发糊"，而不只是验证代码能跑。

GL 侧靠浏览器目视 + 截图：3D 预设机位下分屏对比、掠射角拉动 $N_{\max}$ 观察 Sweeper 侧变锐而 SDF 侧不变、2D 模式与改造前截图一致、字号/相机扫动不触发重建。

## 10. 局限（如实呈现）

- 论文 §4.4 末尾的**剪切变换路线**（对控制点做 2×2 QR 分解 + shear，把足迹变回轴对齐）**不实现**。该路线精度最高，但剪切后曲线不再单调，必须在 shader 里现场重新切分单调段；论文作者本人亦未实现。
- 多窗口是对平行四边形足迹的**分片盒式逼近**，不是真正的各向异性滤波；$N_{\max}$ 封顶后，超过该比值的极端掠射仍会残留模糊。
- 采用**弱透视近似**：一个像素的足迹按平行四边形处理，而严格透视下应为梯形。论文脚注 1 指出这在实际场景中基本总是可接受的。
- 深度测试关闭，故 3D 下只支持**单一共面平面**上的文字；多平面互相遮挡不在范围内。
- SDF/MSDF 侧无各向异性补偿（§5），这是烘焙式方法的固有限制，非实现取舍。

## 11. 风险与缓解

- **导数在图元边缘不准**：`dFdx/dFdy` 按 2×2 quad 计算，字形四边形边缘的 fragment 其 quad 伙伴可能在图元外。属硬件固有行为；因四边形已按 bbox 外扩 `PAD_EM`，边缘处覆盖率本就接近 0，影响可忽略。
- **性能**：$N_{\max}=8$ 叠加复杂 CJK 字形是 8× 曲线循环。缓解：$N$ 由实际 $\rho$ 决定而非恒取 $N_{\max}$，正入射区域自动回落到 1；$N_{\max}$ 可下调；info 读出暴露实时重建/绘制耗时。
- **近平面裁剪**：极端掠射时平面远端可能穿过近平面。GL 的近平面裁剪会正确处理，但 `dFdx` 在被裁剪三角形边缘可能异常；把 near 设小（0.01）并把 dist 下限夹在 0.3 以上规避。
- **2D 回归**：矩阵化与 `fwidth` 化两处都可能悄悄改变 2D 输出。用测试 2 与测试 3 分别在矩阵层和足迹层锁死等价性。
