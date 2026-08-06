# 设计文档：Scanline Sweeper vs SDF/MSDF 字形渲染对比 Demo

- 日期：2026-07-23
- 参考论文：`reference/scanline_sweeper_preprint.pdf`（Scuff3D Rook, *The Scanline Sweeper: A Glyph Rendering Algorithm*, Rook & Possum, 2026）
- 文本抽取：`reference/out.txt`（`pdftotext -layout`）

## 1. 目标

单页 WebGL2 Demo，把同一段文字用三种算法渲染，通过**分屏擦除滑块**逐像素对比：

1. **Scanline Sweeper**（论文算法）——单调二次贝塞尔曲线存入数据纹理，逐像素解析式累加有符号扫掠面积得到覆盖率。分辨率无关，任意放大保持锐利。
2. **经典单通道 SDF**（Green 2007）——烘焙有符号距离场图集。圆角、依赖图集分辨率。
3. **MSDF**（Chlumský 2015）——多通道距离场，保留尖角，但极端放大仍暴露图集分辨率上限。

滑块左侧固定 Sweeper，右侧显示用户选择的 SDF 变体（Single-channel 或 MSDF）。两侧共用同一套 view 变换（平移/缩放/字号），字形在分界线两侧精确对齐，便于对比。

## 2. 非目标（Non-goals）

- 不做字体 hinting（论文 §5.1：小字号竖笔无法对齐像素栅格，如实展示为局限）。
- 不做复杂文字整形（HarfBuzz 级 shaping、连字、阿拉伯语/印度语系整形）。仅左到右按 advance + kerning 布局。
- 不做曲线加速结构（论文 §4.1 的水平分带）——遍历字形全部曲线，符合论文"先上最简版"建议。
- 不实现论文 §4.3 的逐轮廓平均；轮廓自交时对覆盖率做 clamp（见 §11 局限）。
- 不做透视/剪切变换（论文 §4.4 扩展）；仅 2D 正交。

## 3. 技术栈与依赖（全部本地化，运行时不访问 CDN）

- **WebGL2**（必需）：`drawArraysInstanced`（一次 draw call 画全部字形）、`texelFetch`（无过滤读曲线数据）、`RGBA32F` 采样、整型/浮点 instance 属性、`fwidth`。
- **opentype.js** v1.3.4（MIT，~170KB）→ 本地 `vendor/opentype.min.js`。解析 TTF/OTF（TrueType 二次轮廓 + CFF 三次轮廓），提供 glyph path、advanceWidth、kerning、unitsPerEm。
- **内置字体**（开源许可，下载到 `fonts/`）：
  - 衬线：**Tinos**（Apache-2.0，与 Times New Roman 度量兼容）——代表"经典衬线"。
  - 无衬线：**Geist**（OFL-1.1）——呼应论文基准字体。
  - 脚本连写：**Dancing Script**（OFL-1.1）——曲线密集，压测 Sweeper 曲线路径与 SDF 圆角。
  - CJK：**Noto Sans SC** 的 **pyftsubset 子集**（OFL-1.1）——常用汉字（GB2312 一级字 3755 + ASCII + CJK 标点），约 1–3MB。日/韩通过上传实现（在 UI 中注明）。
  - **上传**：任意用户 `.ttf/.otf`。
- 其余全部（渲染、预处理、SDF/MSDF 生成、布局、UI）为自研代码，无其他第三方库。

## 4. 目录结构

Demo 位于 `d:/tmp/Glyph_Render/`（`reference/` 的上级），初始化为 git 仓库：

```
index.html                  UI、canvas、控制面板、模块引入与启动
vendor/opentype.min.js      （vendored）
fonts/*.ttf                 （vendored，含 CJK 子集）
fonts/LICENSES.md           各字体许可与来源
js/gl-utils.js              shader 编译/链接、纹理/缓冲/VAO 助手、单位四边形几何
js/font-loader.js           opentype.js 封装 → 归一化 em 空间轮廓 + 度量
js/layout.js                文本 → 定位 glyph 实例（换行、advance、kerning）
js/sweeper-preprocess.js    轮廓 → 有向单调二次贝塞尔
js/sweeper-renderer.js      曲线打包进数据纹理；instanced draw；sweep 片段着色器
js/sdf-generator.js         解析式 SDF + MSDF 图集生成（共享距离引擎）
js/sdf-renderer.js          instanced draw，采样 SDF/MSDF 图集
js/main.js                  应用控制器：UI 绑定、重建触发、渲染循环、分屏擦除
test/*.mjs                  纯数值模块的 Node 单测（不依赖 GL）
docs/superpowers/specs/     本设计文档
```

模块加载方式：`index.html` 用 `<script type="module">` 引入 `js/main.js`，其余 `js/*.js` 用 ES module `import`。opentype.js 为 UMD，用普通 `<script>` 先引入，暴露全局 `opentype`。

## 5. 坐标系与数据模型

### 5.1 坐标空间

- **font 单位**：opentype.js `glyph.path` 命令的原始坐标，y-up。
- **em 空间**：font 单位 ÷ `unitsPerEm`，保持 **y-up**（font 原生朝向）。所有曲线数据、SDF 图集内容、sweep 数学均在 em 空间。glyph 坐标大致落在 [-0.2, 1.2] 范围。
- **screen/像素空间**：canvas 像素，y-down。em→screen 由 view 变换（uniform）完成，含 y 翻转、字号 `fontSizePx`、缩放 `zoom`、平移 `panPx`、`devicePixelRatio`。

**填充朝向约定**：Sweeper 累加的有符号面积其正负取决于轮廓绕向（TrueType/CFF 约定不同）。实现时对累加覆盖率做一次全局符号归一化，使字形**内部为正**（用 §10 的超采样参考实现在一个测试字形上验证；若反相则整体取负）。

### 5.2 每字形预处理产物（缓存 key = 字体 + glyphId）

- **Sweeper**：有向单调二次曲线列表 `[{p0,p1,p2}, ...]`（em 空间，每条已保证 x、y 双单调）。
- **SDF/MSDF**：给定分辨率下的图集 tile 像素（生成时按需）。
- **公共**：em 空间轮廓包围盒 `bboxMin/bboxMax`、advanceWidth（em）。

### 5.3 布局产物（缓存 key = 文本 + 字体）

glyph 实例数组，每项：`{ emOrigin:vec2 (笔位，y-up), glyphId }`。换行按 `\n`；行高 = 1.2 em（可配）；笔位按 advanceWidth 前进，相邻字用 kerning 修正。

## 6. 数据流与重建契约

| 变化项 | 重建内容 |
|---|---|
| 文本 / 字体 | 重新布局 → 重算 unique glyph 集 → 重建 Sweeper 曲线纹理 + SDF/MSDF 图集 + instance 缓冲 |
| SDF 分辨率 | 仅重建 SDF/MSDF 图集（+ 相关 instance 的 atlasRect） |
| SDF 模式（Single/MSDF） | 若目标图集未缓存则生成，切换采样纹理与 shader 分支 |
| 平移 / 缩放 / 字号 / gamma / SDF spread | **仅改 uniform，不重建任何缓冲/纹理** |

布局以 em 单位存储，交互全走 uniform，满足"仅当显示字符变化时更新纹理"。

## 7. Scanline Sweeper

### 7.1 预处理（`sweeper-preprocess.js`，对应论文 §3.1）

输入：em 空间轮廓命令（moveTo / lineTo / quadraticCurveTo / bezierCurveTo / close）。步骤：

1. **三次 → 二次**：CFF 的三次段用自适应中点细分近似为若干二次段，误差阈值 `ε_em`（默认 1e-3 em），最多递归 N 层。（选简单法而非 Truong 2020 最优法；TrueType 本就是二次，此步只影响 CFF 字体。）
2. **线段升二次**：`lineTo` 段取端点中点作为控制点 `p1`。
3. **删水平段**：`|p0.y - p2.y| < ε` 且为直线的段直接丢弃（不参与算法）。
4. **单调化**：对每条二次段，求 x 与 y 方向的极值参数 `t*`（`t = (p0-p1)/(p0-2p1+p2)`，若在 (0,1) 内），在这些 t 处用 de Casteljau 切分，直到每个子段在 [0,1] 上 **x、y 双单调**（每维最多一个极值 → 最多切 2 次）。
5. 输出有向控制点 `p0,p1,p2`（顺序即曲线方向，编码符号）。确保 `p0,p2` 为端点。

预处理产物即可用 AABB（首末控制点）做剔除，且任意曲线-直线交点在域内至多一个根（论文保证）。

### 7.2 曲线数据纹理

- 格式 `RGBA32F`，2D，宽 `W`（默认 1024），高按需增长。NEAREST 采样，`texelFetch` 读取。
- 每条曲线占 **2 个 texel**：`texel0 = (p0.x, p0.y, p1.x, p1.y)`、`texel1 = (p2.x, p2.y, 0, 0)`。
- 全局曲线顺序 = 按 unique glyph 顺序拼接。每字形记录 `curveStartTexel`（= 全局曲线序号 × 2）与 `curveCount`，随 instance 传入。
- texel 序号 `idx` → 纹理坐标 `(idx % W, idx / W)`。

### 7.3 片段着色器（GLSL ES 3.00，移植论文 §8）

- `evaluate_bezier`、`intersect_monotonic`（[out.txt:671](../../reference/out.txt#L671)）、`scanline_sweep`（[out.txt:730](../../reference/out.txt#L730)）逐行移植为 GLSL。HLSL `mad/lerp/saturate/sign` → GLSL `fma`（或乘加）/`mix`/`clamp01`/`sign`。
- 每 fragment：
  1. 取 glyph-local em 坐标 `fragEm`（varying）。
  2. 窗口 `size = vec2(emPerPixel)`（uniform，正交下精确 = `1/(fontSizePx*zoom*dpr)`，方形像素）；`offset = fragEm - 0.5*size`（像素足迹以 fragment 为中心）。
  3. `float area = 0; for (i in 0..curveCount) { 读 p0,p1,p2; area += scanline_sweep(size, offset, p0,p1,p2); }`
  4. `coverage = clamp(sign_norm * area / (size.x*size.y), 0, 1)`；`alpha = pow(coverage, 1/gamma)`；输出 `(textColor.rgb, alpha)`。
- `curveCount` 循环上界用一个常量最大值 + 动态 `break`（WebGL2 需常量循环边界；用 `for (int i=0;i<MAX;i++){ if(i>=count) break; ... }`）。`MAX` 取足够大（如 512）。

## 8. SDF 与 MSDF 生成（`sdf-generator.js`）

一套 CPU **距离引擎**服务两者（即"从贝塞尔曲线算 SDF"的 ground-truth 路线；不用 WASM msdfgen 以免笨重脆弱的构建，也不用 raster+EDT 因其做不了 MSDF）。

### 8.1 边表示

把每字形轮廓展开为**边**序列（保留轮廓闭合与角点信息）。为简化距离计算，把二次/三次段**细分为线段**（每曲线 8–16 段，子像素误差）。每条边记录：起点、终点、方向、所属轮廓、以及（MSDF 用）颜色通道掩码。

### 8.2 图集布局

- tile 尺寸 = `res + 2*pad`（`res` ∈ {16,24,32,48,64}，`pad` 默认 4 texel）。tile 覆盖 glyph em bbox 外扩 `pad/res` 的区域。
- 简单行式（shelf）打包进图集纹理：SDF 用 `R8`，MSDF 用 `RGB8`（存 `RGBA8`，A 备用）。
- 每字形记录归一化 `atlasRect (u0,v0,u1,v1)`，随 instance 传入。
- 距离编码：`store = clamp(0.5 + signedDist_em / (2*spread_em), 0, 1)`；`spread_em` 由 pad 对应的 em 宽度决定。**符号约定：字形内部 `signedDist > 0`**（→ `store > 0.5`），内外统一贯穿 SDF/MSDF 与 shader 解码。

### 8.3 单通道 SDF

每 texel 中心（em）：对全部边求无符号最近距离；符号用**非零绕数**点在多边形内测试，内部取正（见 §8.2 约定）。写入编码值。

### 8.4 MSDF（对应 msdfgen `edgeColoringSimple` + 伪距离）

1. **角点检测**：轮廓上相邻边切线夹角 > 阈值（默认 3°，即非平滑衔接）处为角点。
2. **边着色**：颜色为通道掩码 `YELLOW=R+G(110)`、`MAGENTA=R+B(101)`、`CYAN=G+B(011)`。沿轮廓遍历，在每个角点切换颜色，保证相邻边**恰好共享一个通道**。特例：轮廓 0 角点（全平滑环）→ 单色 teardrop 处理；1 角点 → 在该角点分两半、赋两色。
3. **逐通道伪距离**：对每个通道 c，在所有含 c 的边中，取**真实距离**最小者，用其**伪距离**写入通道 c。线段伪距离：投影参数 `t∈[0,1]` 时为垂直有符号距离；`t<0/t>1` 时为到该边**无限支撑线**的有符号距离（越过端点仍沿直线延伸——这正是保尖角的关键）。符号由叉积朝向定。
4. shader 侧 `d = median(r,g,b)` 后与 SDF 同样解码。

### 8.5 生成性能

`O(texel 数 × 边数)`/字形。`res≤64`、数十个 unique glyph 下同步可接受，带进度条。CJK 大量字符时可能卡顿——**Web Worker 异步生成**列为可选升级项（先做同步版）。

### 8.6 渲染侧解码与抗锯齿（`sdf-renderer.js`）

片段着色器：SDF 直接采样，MSDF 先 `m = median(r,g,b)`。解码回 em 距离 `d_em = (m - 0.5) * 2 * spread_em`。抗锯齿用屏幕空间过渡：`aa = 0.5 * emPerPixel`（半像素的 em 宽度），`alpha = clamp(0.5 + d_em / (2*aa), 0, 1)`（约一像素线性边缘）；`spread` slider 调节软硬。emPerPixel 与 Sweeper 用同一 uniform，保证两侧边缘过渡尺度一致、可比。

## 9. Instanced 渲染与打包

- **基础几何**：单位四边形 4 顶点（triangle strip），corner ∈ {(0,0),(1,0),(0,1),(1,1)}。
- **instance 缓冲**（单个 `Float32Array`，stride 12 floats；小整数 `curveStart/curveCount` 以 float 存储，<2^24 精确）：
  `[emOrigin.xy, emBBoxMin.xy, emBBoxMax.xy, curveStart, curveCount, atlasRect.xyzw]`
- **顶点着色器**：`localEm = mix(emBBoxMin - padEm, emBBoxMax + padEm, corner)`；`worldEm = localEm + emOrigin`；`clip = viewMatrix * worldEm`（em→screen，含 y 翻转/字号/缩放/平移/dpr）。varying：Sweeper 传 `localEm`；SDF 传图集 `uv = mix(atlasRect.xy, atlasRect.zw, cornerRemap)`。
- **一次 draw call**：每种算法 `gl.drawArraysInstanced(TRIANGLE_STRIP, 0, 4, instanceCount)` 画完全部字形。
- **分屏擦除**：同一 canvas，`gl.enable(SCISSOR_TEST)`；Sweeper 限制到分界线左侧矩形，SDF 限制到右侧；两者同 view 变换 → 跨缝对齐。分界线与手柄用 HTML/CSS 覆盖层绘制，可拖动改变 `sliderX`。

## 10. UI 与交互（`index.html` + `main.js`）

控制面板：
- **字体**下拉（内置 Tinos/Geist/Dancing Script/Noto Sans SC 子集 + "上传…" file input）。
- **SDF 分辨率**下拉（16/24/32/48/64 px-em）。
- **SDF 模式**切换（Single-channel / MSDF）——决定滑块右侧显示哪种。
- **字号** slider（px）。
- **gamma** slider（Sweeper）。
- **SDF spread** slider（可选，影响 AA 软硬）。
- 画布**拖拽平移** + **滚轮缩放** + **重置视图**按钮。
- **文本框**（多行 textarea，Latin 或 CJK）。
- **信息读出**：unique glyph 数、总曲线数、图集尺寸、纹理显存估算、重建耗时——把"内存/质量"权衡量化，服务对比叙事。

## 11. 已知局限（如实呈现，不隐藏）

- 轮廓自交仅对覆盖率 clamp（论文 §4.3 未实现逐轮廓平均）——极少数自交字形可能轻微偏差。
- 无 hinting——小字号竖笔不对齐像素栅格（论文 §5.1）。
- 简单布局——advance + kerning，无复杂 shaping/连字。
- 无曲线加速结构——逐 fragment 遍历字形全部曲线；复杂 CJK 字形在大面积覆盖时片段着色较重（demo 尺度可接受）。
- 内置 CJK 仅简体子集；日/韩及生僻字通过上传字体解决。

## 12. 验证计划（数值核心 TDD）

Node 单测（`test/*.mjs`，不依赖 GL），实现前先写：

1. **预处理**（`sweeper-preprocess`）：断言输出每段严格 x、y 单调；在密集采样点上还原原始轮廓，误差 < 阈值。
2. **sweep 参考实现**（JS 移植 `scanline_sweep` + `intersect_monotonic`）：
   - 单位方块轮廓：窗口完全在内 → 覆盖 ≈1；完全在外 → ≈0；恰跨边 → ≈0.5。
   - 真实字形（如 'A'、'e'、'口'）：与 **16× 超采样非零绕数光栅化** ground truth 比较，平均绝对覆盖误差 < 阈值（如 0.02）。
3. **距离引擎 / MSDF**：median 的内外符号正确；已知 90° 角点在低分辨率（res=16）下，MSDF 的角点锐度（沿角平分线的距离场梯度一致性）明显优于单通道 SDF——量化断言。

GL 渲染结果靠肉眼 + 截图验证（分屏对比、放大观察 SDF 圆角/MSDF 尖角/Sweeper 锐利、字号扫动无重建）。

## 13. 风险与缓解

- **填充符号约定**：TrueType(CW) vs CFF(CCW) 绕向不同 → 用参考实现验证并全局翻转符号（§5.1）。
- **RGBA32F 可用性**：WebGL2 核心支持 NEAREST 采样 32F 纹理；启动时检测 `EXT_color_buffer_float` 仅在需渲染到 float 时；本方案只采样，无需。仍在初始化做能力检测与降级提示。
- **CJK 子集构建**：`pyftsubset` 裁剪；若某字体无对应 glyph 则跳过并提示上传。
- **MSDF 边着色正确性**：以 msdfgen 简单着色算法为准，单测覆盖角点相邻通道约束；错误表现为角点处颜色伪影，可视化排查。
