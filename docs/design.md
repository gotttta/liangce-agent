# DRAM 缺陷量测 Agent - 设计文档

## 1. 项目定位

一个面向 DRAM 显微/SEM 图像的对话式缺陷量测 Agent。用户提供缺陷参考图、缺陷描述和未经标注的待测原图，Agent 通过多轮交互生成并调整图像处理流程，对缺陷进行自动标注和量测。

**核心原则：**
- 用户不需要懂 CV 术语，用看图反馈驱动调整
- 每次处理后展示标注图和量测结果，用户只判断「多了、少了、边界准不准」
- MVP 阶段优先打通闭环，后续逐步把稳定逻辑沉淀为可复用模块
- 量测结果必须带单位、置信度/状态说明和失败原因

---

## 2. 输入与输出

### 2.1 输入

| 输入 | 必需 | 作用 |
|------|------|------|
| handbook/reference 图 | 是 | 说明缺陷长什么样、希望关注哪类结构 |
| 缺陷文字描述 | 是 | 描述缺陷名称、现象、希望量测的指标 |
| target/wafer 原图 | 是 | 实际需要自动标注和量测的未标注图片 |
| 标定信息 | 可选 | 像素尺寸、倍率或 scale bar，用于换算物理单位 |
| 用户修正标记 | 可选 | 用户圈出漏检、误检或边界不准区域 |

> 如果没有标定信息，系统只输出 pixel 单位；如果只有 reference 图，没有 target 图，则只能做算法演示，不能作为真实量测结果。

### 2.2 输出

每次运行输出一个结果目录：

```
outputs/<timestamp>_<defect_type>/
├── algorithm.py              # MVP 阶段生成的受限脚本
├── strategy.json             # Agent 决策记录
├── result_annotated.png      # 标注叠加图
├── mask.png                  # 缺陷 mask
└── measurements.json         # 量测结果
```

`measurements.json` 至少包含：

```json
{
  "defect_type": "bridge",
  "measurement_type": "area",
  "unit": "pixel",
  "status": "ok",
  "results": [
    {
      "id": 1,
      "area": 1832,
      "bbox": [120, 88, 54, 31],
      "confidence": 0.74
    }
  ],
  "notes": ["No calibration provided; physical unit is unavailable."]
}
```

---

## 3. 核心工作流

```
用户上传 reference 图 + target 原图 + 缺陷描述
           │
           ▼
┌─────────────────────────────────────┐
│ Step 1: 多模态 LLM 理解              │
│ - 缺陷类型                           │
│ - 目标结构                           │
│ - 推荐量测项                         │
│ - 输出 strategy.json                 │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ Step 2: 生成/选择处理流程             │
│ MVP: 生成受限 Python 脚本             │
│ 稳定后: 调用固定 core 模块             │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ Step 3: 执行分割 + 量测               │
│ SAM prompt 分割 + 传统 CV 测量         │
│ 或纯传统 CV baseline                  │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ Step 4: 前端展示结果                  │
│ 标注图 + mask + 量测数值 + 状态说明    │
└──────────────────┬──────────────────┘
                   ▼
     用户反馈: 漏了/多了/边界不准/量错了
                   │
                   ▼
          Agent 调整 strategy 并重跑
```

---

## 4. DRAM v1 量测范围

v1 不追求覆盖所有 DRAM defect，先覆盖最容易形成闭环、可解释、可由 2D 图像支持的量测项。

| 类型 | 典型对象 | 量测项 | v1 优先级 |
|------|----------|--------|----------|
| Area / Size | 残留、污染、颗粒、局部异常区域 | 面积、外接框、长度、宽度、长宽比 | P0 |
| Count / Density | 多颗粒、多缺陷区域 | 数量、单位面积密度、区域占比 | P0 |
| CD / Space | wordline、bitline、阵列线条 | 线宽、间距、局部变窄/变宽 | P1 |
| Bridge / Open | 桥连、断线、缺口 | 桥连宽度、断口间隙、异常长度 | P1 |
| Contact / Hole | contact、via、storage node contact | 孔径、圆度、孔面积、中心偏移 | P1 |
| Overlay / Alignment | 多层结构 | X/Y 偏移、contact-to-line offset | P2 |
| Roughness | 线边、轮廓 | LER、LWR、轮廓偏差 | P2 |

### v1 MVP 建议

第一版优先实现：

1. Area / Size
2. Count / Density
3. Bridge / Open 的简单版本

CD、Contact、Overlay、Roughness 对图像质量、标定、结构假设要求更高，作为第二阶段扩展。

---

## 5. Agent 策略与代码生成方式

### 5.1 MVP 阶段

MVP 阶段允许 LLM 生成受限 Python 脚本，用来快速验证不同缺陷类型的处理闭环。

约束：
- 脚本只在 `outputs/<run_id>/` 下读写文件
- 脚本必须输出 `result_annotated.png`、`mask.png`、`measurements.json`
- 禁止访问网络、系统命令和非项目目录
- 每次生成的脚本同时保存 `strategy.json`，记录使用的方法和参数

### 5.2 稳定阶段

当多个生成脚本出现重复逻辑后，将通用能力沉淀为模块：

```
core/
├── preprocessing.py          # 灰度化、去噪、对比度增强、背景校正
├── segmentation.py           # SAM prompt、阈值、边缘、连通域筛选
├── measurement/
│   ├── area.py               # 面积、外接框、长宽比
│   ├── count.py              # 数量、密度、区域占比
│   ├── cd.py                 # 线宽、间距、局部异常
│   ├── contact.py            # 孔径、圆度、中心偏移
│   └── overlay.py            # 层间偏移
└── visualization.py          # mask 叠加、轮廓、量测标注
```

此时 LLM 不再直接写完整算法，而是输出结构化策略：

```json
{
  "defect_type": "particle",
  "target_structure": "array region",
  "measurement_type": "area_count",
  "preprocess": {
    "contrast": "enhance",
    "denoise": "light"
  },
  "segmentation": {
    "method": "sam_prompt",
    "sensitivity": 0.65
  },
  "measurement": {
    "min_area_px": 20,
    "unit": "pixel"
  }
}
```

---

## 6. 技术选型

| 模块 | 选择 | 原因 |
|------|------|------|
| 多模态理解 | Doubao Seed 2.0 Pro | 识别 reference/target 图中的缺陷形态，生成结构化策略 |
| 缺陷分割 | SAM 轻量变体 + prompt | 适合交互式修正；先作为通用分割能力 |
| 传统 CV 测量 | OpenCV + scikit-image | 量测逻辑确定、可复现、容易解释 |
| 前端交互 | Gradio ImageEditor | 支持图片查看、图层叠加和用户标记 |
| Agent 编排 | Claude Code 风格对话引擎 | 负责多轮沟通、策略调整和运行记录 |

### 6.1 技术边界

- SAM 对 DRAM/SEM 图像不保证开箱即用，需要用户 feedback 和传统 CV 兜底
- 没有标定信息时，不能输出 um/nm 等物理单位
- 单张 2D top-down 图通常无法可靠量测高度、深度、体积
- Overlay 需要能同时识别两层结构，或需要设计/版图/参考层信息
- Roughness 对分辨率、边缘质量和噪声非常敏感，v1 不作为主目标

---

## 7. 前端功能

| 功能 | 说明 |
|------|------|
| 图片上传 | 上传 reference 图、target 原图、可选 scale 信息 |
| 图片查看 | 原图、mask、标注叠加图之间切换 |
| 标记工具 | 用户圈出漏检、误检、边界不准区域 |
| 反馈类型 | 选择「少圈了」「圈多了」「边界不准」「量测类型不对」 |
| 结果展示 | 显示面积/数量/CD 等结果、单位、状态和说明 |
| 导出 | 下载标注图、mask、measurements.json 和策略记录 |

---

## 8. 交互语言原则

对用户展示时避免直接使用 CV 术语。

| 技术说法 | 用户说法 |
|----------|----------|
| 阈值 | 灵敏度 |
| 形态学闭运算 | 把断开的地方连起来 |
| 连通域过滤 | 去掉太小的噪点 |
| 骨架化 | 沿着中线量宽度 |
| mask | 圈出来的区域 |

示例：

- 「这次圈得多了还是少了？」
- 「哪些地方不应该被圈进去？」
- 「这条线是想量宽度，还是想量断开的间隙？」
- 「当前没有比例尺，所以结果先用像素表示。」

---

## 9. 数据流

```
reference 图 + 缺陷描述 ──► 多模态理解 ──► defect strategy
         target 原图 ───────────────┘
                                      │
                                      ▼
                         生成脚本或调用 core 模块
                                      │
                                      ▼
                          分割 mask + 量测 JSON
                                      │
                                      ▼
                         Gradio 展示标注图和结果
                                      │
                                      ▼
                         用户标记修正区域和反馈类型
                                      │
                                      ▼
                         更新 strategy 并重跑
```

---

## 10. 失败处理

系统需要显式告诉用户为什么这次结果不可靠，而不是只输出空结果。

| 场景 | 系统行为 |
|------|----------|
| 没有 target 原图 | 提示只能做 reference 演示，不能做真实量测 |
| 没有标定信息 | 输出 pixel 单位，并提示无法换算物理尺寸 |
| 分割置信度低 | 展示候选结果，请用户圈出正确区域 |
| 缺陷与背景太像 | 提示需要更清晰图片或用户手动画初始区域 |
| Overlay 特征不足 | 提示需要两层结构、设计参考或更明确的参考点 |
| 结果为空 | 输出空 JSON + 失败原因 + 建议下一步 |

---

## 11. 后续扩展

- 批量处理多张 target 图
- wafer/die/region 级统计和 defect density heatmap
- 接入版图或模板，支持 array 周期结构对齐
- 支持 Contact/Hole、Overlay、LER/LWR 的稳定量测
- 积累样本后训练/微调专用缺陷分割模型
