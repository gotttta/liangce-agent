# UI 重构建议：Claude Code 风格改进

## 现状评估

你的 UI 已经有很好的 Claude Code / Codex 风格基础：

✅ **做得好的地方**：
1. 左侧边栏 260px 固定宽度，颜色使用 `#f7f7f5`（Codex 标准）
2. 聊天主区域居中 780px，消息气泡样式接近 Claude
3. CSS 变量命名规范（`--codex-*`）
4. 使用 Inter 字体
5. 响应式设计考虑了移动端

❌ **需要改进的地方**：
1. **布局复杂度过高**：2700+ 行代码，CSS 和逻辑混在一起
2. **组件职责不清**：一个文件包含 UI、状态管理、Agent 调用
3. **交互流程不够流畅**：画布、按钮、消息混在一起
4. **视觉层级不够清晰**：缺少 Claude Code 的"任务卡片"概念

---

## 核心改进方向

### 1. 采用 Claude Code 的"任务流"交互模式

**现状**：用户需要手动点击"新建任务" → 上传图片 → 输入描述 → 等待结果 → 点击按钮

**建议**：模仿 Claude Code 的自然流式对话

```
用户输入：
  上传图片 + "提取颗粒缺陷"

Agent 回复：
  ┌─────────────────────────────────────┐
  │ 🔄 正在执行任务                        │
  │ ✓ 视觉理解完成                         │
  │ ⏳ 生成算法中...                       │
  └─────────────────────────────────────┘

Agent 回复（完成后）：
  ┌─────────────────────────────────────┐
  │ ✓ 任务完成                            │
  │                                      │
  │ [标注结果图]                          │
  │                                      │
  │ 检测到 12 个区域，总面积 3456 px²     │
  │                                      │
  │ 【✓ 标注准确】 【✗ 有误，修改】        │
  └─────────────────────────────────────┘
```

**实现建议**：
- 将当前的 `next_actions` 区域移入聊天消息卡片内
- 用 Gradio 的 `gr.Row` + `gr.Button` 实现内嵌按钮
- 参考 Claude Code 的"工具调用卡片"样式

---

### 2. 简化画布交互流程

**现状**：用户点击"有漏标或错标" → 画布出现在聊天区域下方 → 用户涂抹 → 发送

**问题**：
- 画布突然出现在底部，视觉跳跃大
- 红色/绿色两个 Tab 切换繁琐

**建议**：模仿 Codex 的"工具面板"

```css
/* 画布作为浮动工具面板，覆盖在聊天区域右侧 */
#annotation-feedback-canvas {
    position: fixed;
    right: 24px;
    top: 80px;
    bottom: 24px;
    width: 380px;
    z-index: 100;
    background: #ffffff;
    border: 1px solid var(--codex-border);
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}
```

**交互流程**：
1. 用户点击"有误，修改" → 画布从右侧滑入
2. 工具栏固定在画布顶部：`[红色笔] [绿色笔] [橡皮擦] [清空]`
3. 底部固定提交按钮：`[取消] [提交修改]`
4. 画布内嵌缩略图 + 放大镜工具

**参考实现**：
```python
feedback_canvas = gr.Column(
    visible=False,
    elem_id="annotation-feedback-canvas",
    elem_classes=["floating-panel"],
)
with feedback_canvas:
    gr.HTML('<div class="panel-title">标记需要修改的区域</div>')
    with gr.Row(elem_id="annotation-brush-toolbar"):
        brush_red = gr.Radio(["红色笔"], value="红色笔", label="误检删除")
        brush_green = gr.Radio(["绿色笔"], label="漏检补充")
    feedback_editor = gr.ImageEditor(...)
    with gr.Row():
        cancel_btn = gr.Button("取消")
        submit_btn = gr.Button("提交修改", variant="primary")
```

---

### 3. 重构文件结构

**现状**：`ui/annotation_app.py` 2700+ 行，包含：
- CSS 样式（1000+ 行）
- UI 组件定义（800+ 行）
- 业务逻辑（900+ 行）

**建议**：拆分为独立模块

```
ui/
├── app.py                  # 入口，只负责启动
├── annotation_app.py       # 主 UI 组装（200 行）
├── components/
│   ├── chat.py            # 聊天区域组件
│   ├── sidebar.py         # 左侧边栏
│   ├── canvas.py          # 画布工具面板
│   └── composer.py        # 输入框组件
├── handlers/
│   ├── agent.py           # Agent 调用逻辑
│   ├── feedback.py        # 用户反馈处理
│   └── task.py            # 任务管理
├── styles/
│   ├── base.css           # 基础样式
│   ├── chat.css           # 聊天区域样式
│   ├── sidebar.css        # 侧边栏样式
│   └── canvas.css         # 画布样式
└── utils/
    ├── formatters.py      # 消息格式化
    └── state.py           # 状态管理辅助
```

**示例拆分**：

```python
# ui/components/chat.py
def build_chat_component():
    with gr.Column(elem_id="annotation-chat"):
        gr.HTML('<div id="annotation-welcome">...</div>')
        chatbot = gr.Chatbot(...)
        result_image = gr.Image(...)
        measurements = gr.Dataframe(...)
    return {
        "chatbot": chatbot,
        "result_image": result_image,
        "measurements": measurements,
    }

# ui/annotation_app.py（重构后）
from ui.components.chat import build_chat_component
from ui.components.sidebar import build_sidebar_component
from ui.components.canvas import build_canvas_component

def build_annotation_app():
    with gr.Blocks(...) as app:
        with gr.Row(elem_id="annotation-shell"):
            sidebar = build_sidebar_component()
            chat = build_chat_component()
            canvas = build_canvas_component()

        # 事件绑定
        setup_events(sidebar, chat, canvas)

    return app
```

---

### 4. 优化 CSS 架构

**现状**：1000+ 行 CSS 写在 Python 字符串里

**建议**：提取到独立 CSS 文件，按组件分层

```css
/* styles/base.css */
:root {
    --codex-sidebar: #f7f7f5;
    --codex-hover: #ececea;
    --codex-active: #e7e7e4;
    --codex-border: #e5e5e2;
    --codex-ink: #1f1f1f;
    --codex-muted: #737373;

    /* 新增：尺寸变量 */
    --sidebar-width: 260px;
    --chat-max-width: 780px;
    --composer-max-width: 720px;
    --message-radius: 18px;
    --panel-radius: 12px;
}

/* styles/chat.css */
#annotation-chatbot .message.bot {
    /* 移除固定宽度，改用 flex 自适应 */
    width: auto;
    max-width: 100%;
    padding: 2px 0;
}

/* 新增：任务卡片样式（模仿 Claude Code） */
.task-card {
    margin: 16px 0;
    padding: 16px 20px;
    border: 1px solid var(--codex-border);
    border-radius: var(--panel-radius);
    background: #fafafa;
}

.task-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    font-size: 13px;
    font-weight: 600;
}

.task-card-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 12px;
    background: #e8f5e9;
    color: #2e7d32;
    font-size: 11px;
}

/* styles/canvas.css */
.floating-panel {
    position: fixed;
    right: 24px;
    top: 80px;
    bottom: 24px;
    width: 380px;
    z-index: 100;
    background: #ffffff;
    border: 1px solid var(--codex-border);
    border-radius: var(--panel-radius);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    transform: translateX(420px);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.floating-panel.visible {
    transform: translateX(0);
}
```

**加载方式**：

```python
# ui/styles/__init__.py
from pathlib import Path

def load_styles():
    styles_dir = Path(__file__).parent
    css_files = ["base.css", "chat.css", "sidebar.css", "canvas.css"]
    return "\n".join(
        (styles_dir / name).read_text(encoding="utf-8")
        for name in css_files
    )

# ui/annotation_app.py
from ui.styles import load_styles

def build_annotation_app():
    with gr.Blocks(css=load_styles()) as app:
        ...
```

---

### 5. 改进消息展示逻辑

**现状**：Agent 执行过程通过 `_format_agent_process` 生成 Markdown

**问题**：
- 执行过程和结果混在一起
- 缺少 Claude Code 的"工具调用卡片"视觉区分

**建议**：使用结构化消息格式

```python
# ui/utils/formatters.py
def format_task_card(state):
    """生成 Claude Code 风格的任务卡片"""
    status = state.get("agent_status")
    status_config = {
        "waiting_for_acceptance": ("✓ 任务完成", "#22a06b", "#e8f5e9"),
        "waiting_for_feedback": ("⏳ 等待反馈", "#f59e0b", "#fef3c7"),
        "accepted": ("✓ 已确认", "#22a06b", "#e8f5e9"),
    }
    label, color, bg = status_config.get(status, ("• 处理中", "#737373", "#f3f4f6"))

    measurements = state.get("measurements", {}).get("summary", {})
    count = measurements.get("count", 0)
    area = measurements.get("total_area", 0)

    return f"""
<div class="task-card">
    <div class="task-card-header">
        <span>算法执行结果</span>
        <span class="task-card-status" style="background: {bg}; color: {color};">
            {label}
        </span>
    </div>
    <div class="task-card-metrics">
        <div class="metric"><strong>{count}</strong> 个区域</div>
        <div class="metric"><strong>{area:.0f}</strong> px²</div>
    </div>
</div>
"""

# ui/handlers/agent.py
def run_chat_agent(...):
    # ...
    messages.append({
        "role": "assistant",
        "content": format_task_card(state),
    })
    messages.append({
        "role": "assistant",
        "content": (state["annotated_image_path"], "点击放大标注结果"),
    })
```

**效果**：
```
┌───────────────────────────────────┐
│ 算法执行结果  ✓ 任务完成           │
│                                   │
│ 12 个区域    3456 px²             │
└───────────────────────────────────┘

[标注结果图]
```

---

### 6. 优化按钮交互逻辑

**现状**：按钮分散在多个区域，职责重叠

```python
# 当前结构
next_actions:           # 主操作区
    - accept_button
    - continue_button
    - exit_button
    - show_mask_button
    - show_measurements_button

feedback_canvas:        # 画布区
    - clear_feedback_button

composer:               # 输入区
    - attach_button
    - send_button
```

**建议**：按 Claude Code 的"上下文动作"模式重组

```python
# 1. 结果卡片内的快捷操作（内嵌在消息中）
def create_result_actions(state):
    return gr.Row([
        gr.Button("✓ 标注准确", variant="primary"),
        gr.Button("✗ 有误，修改", variant="secondary"),
        gr.Button("查看详情 ▼", variant="secondary"),
    ], elem_id="inline-result-actions")

# 2. 画布工具栏（浮动面板顶部）
def create_canvas_toolbar():
    return gr.Row([
        gr.Radio(["误检（红）", "漏检（绿）"], value="误检（红）"),
        gr.Button("橡皮擦"),
        gr.Button("清空"),
    ], elem_id="canvas-toolbar")

# 3. 主输入框操作（底部固定）
def create_composer():
    with gr.Row(elem_id="annotation-composer"):
        attach_btn = gr.UploadButton("+")
        prompt = gr.Textbox(...)
        send_btn = gr.Button("↑")
```

**优先级**：
- **P0（高频）**：标注准确 / 有误修改 → 内嵌在结果卡片
- **P1（中频）**：查看 Mask / 量测 → 折叠菜单
- **P2（低频）**：退出任务 → 移到侧边栏底部

---

### 7. 实现流式状态更新

**现状**：`run_chat_agent_stream` 使用队列 + 线程轮询

**问题**：
- 更新频率固定，无法根据节点动态调整
- 进度信息和最终结果分离

**建议**：参考 Claude Code 的"思考过程"展示

```python
# ui/handlers/agent.py
def run_chat_agent_stream(...):
    # 初始消息
    yield initial_state_with_loading_card()

    # 实时更新执行过程
    for event in agent_execution_events:
        progress_card = format_progress_card(event)
        working_messages[-1]["content"] = progress_card
        yield (working_messages, ...)

    # 完成后替换为结果卡片
    final_messages[-1]["content"] = format_task_card(final_state)
    yield (final_messages, ...)

def format_progress_card(event):
    """实时进度卡片（可折叠）"""
    return f"""
<details open class="progress-card">
    <summary>
        <span class="progress-icon">⏳</span>
        <span class="progress-label">正在执行算法</span>
        <span class="progress-time">{event['elapsed']:.1f}s</span>
    </summary>
    <div class="progress-steps">
        {''.join(format_step(s) for s in event['steps'])}
    </div>
</details>
"""
```

**效果**：
```
▼ ⏳ 正在执行算法  3.2s
  ✓ 视觉理解完成     1.2s
  ✓ 生成算法完成     0.8s
  ⏳ 执行 Pipeline    1.2s / 估计剩余 0.5s
```

---

### 8. 移动端适配改进

**现状**：`@media (max-width: 820px)` 隐藏侧边栏

**建议**：参考 Claude Code 移动端的"抽屉式侧边栏"

```css
/* styles/responsive.css */
@media (max-width: 820px) {
    #annotation-sidebar {
        position: fixed;
        top: 0;
        left: 0;
        bottom: 0;
        width: 280px;
        transform: translateX(-100%);
        transition: transform 0.3s;
        z-index: 200;
    }

    #annotation-sidebar.open {
        transform: translateX(0);
    }

    .sidebar-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.5);
        z-index: 199;
    }

    .sidebar-overlay.visible {
        display: block;
    }

    /* 顶部添加菜单按钮 */
    #mobile-menu-button {
        display: flex;
        position: fixed;
        top: 12px;
        left: 12px;
        z-index: 201;
    }
}
```

```python
# ui/components/sidebar.py
def build_sidebar_component():
    with gr.Column(elem_id="annotation-sidebar"):
        gr.HTML('<button id="mobile-menu-button">☰</button>')
        # ... 现有侧边栏内容

    gr.HTML(
        '<div class="sidebar-overlay" onclick="closeSidebar()"></div>',
        visible=False,
    )
```

---

## 优先级实施计划

### Phase 1: 结构重构（1-2 天）
- [ ] 拆分 CSS 到独立文件（`styles/`）
- [ ] 提取组件到独立模块（`components/`）
- [ ] 分离业务逻辑到 `handlers/`
- [ ] 验证功能完整性

### Phase 2: 交互优化（2-3 天）
- [ ] 实现任务卡片样式
- [ ] 将按钮移入消息卡片
- [ ] 改造画布为浮动面板
- [ ] 优化流式状态更新

### Phase 3: 视觉细节（1 天）
- [ ] 统一圆角、间距、颜色
- [ ] 添加过渡动画
- [ ] 优化移动端适配
- [ ] 补充暗色模式支持（可选）

---

## 参考资源

### Claude Code 的设计特点
1. **信息层级清晰**：主对话 > 工具调用卡片 > 详细信息折叠
2. **操作即时反馈**：按钮点击 → 立即显示加载状态 → 流式更新
3. **上下文相关**：只在需要时显示相关操作
4. **极简布局**：左侧固定 260px，主区域居中 700-800px，右侧可选面板

### Gradio 最佳实践
- 使用 `gr.State` 管理会话状态，避免全局变量
- 用 `elem_id` 而非 `elem_classes` 定义唯一组件
- 流式输出用 `yield`，配合 `show_progress="hidden"`
- 复杂交互用 `.then()` 链式调用，保持逻辑清晰

---

## 关键代码示例

### 示例 1：任务卡片组件

```python
# ui/components/task_card.py
def render_task_card(state, show_actions=True):
    status = state.get("agent_status")
    measurements = state.get("measurements", {}).get("summary", {})

    card_html = f"""
    <div class="task-card">
        <div class="task-card-header">
            <span class="task-icon">🎯</span>
            <span class="task-title">算法执行结果</span>
            <span class="task-status status-{status}">{_status_label(status)}</span>
        </div>
        <div class="task-metrics">
            <div class="metric">
                <div class="metric-value">{measurements.get('count', 0)}</div>
                <div class="metric-label">检测区域</div>
            </div>
            <div class="metric">
                <div class="metric-value">{measurements.get('total_area', 0):.0f}</div>
                <div class="metric-label">总面积 (px²)</div>
            </div>
        </div>
    </div>
    """

    components = [gr.HTML(card_html)]

    if show_actions:
        with gr.Row(elem_id="task-actions"):
            components.extend([
                gr.Button("✓ 标注准确", variant="primary"),
                gr.Button("✗ 有误，修改", variant="secondary"),
                gr.Button("查看详情", variant="secondary"),
            ])

    return components
```

### 示例 2：浮动画布面板

```python
# ui/components/canvas.py
def build_canvas_panel():
    with gr.Column(
        visible=False,
        elem_id="annotation-canvas-panel",
        elem_classes=["floating-panel"],
    ) as panel:
        with gr.Row(elem_id="canvas-header"):
            gr.HTML('<div class="panel-title">标记需要修改的区域</div>')
            close_btn = gr.Button("×", elem_id="canvas-close")

        with gr.Row(elem_id="canvas-toolbar"):
            brush_mode = gr.Radio(
                ["误检（红色）", "漏检（绿色）"],
                value="误检（红色）",
                label="画笔模式",
            )
            clear_btn = gr.Button("清空画板", variant="secondary")

        editor = gr.ImageEditor(
            label="",
            type="filepath",
            height=480,
            sources=[],
            transforms=(),
            brush=gr.Brush(default_size=12, colors=["#ff3b30", "#22c55e"]),
        )

        feedback_text = gr.Textbox(
            placeholder="描述具体问题，例如：右上角漏了一个颗粒",
            lines=2,
        )

        with gr.Row(elem_id="canvas-footer"):
            cancel_btn = gr.Button("取消", variant="secondary")
            submit_btn = gr.Button("提交修改", variant="primary")

    return {
        "panel": panel,
        "editor": editor,
        "feedback_text": feedback_text,
        "brush_mode": brush_mode,
        "close_btn": close_btn,
        "clear_btn": clear_btn,
        "cancel_btn": cancel_btn,
        "submit_btn": submit_btn,
    }
```

---

## 总结

你的 UI 已经有了很好的 Claude Code 基础，主要改进方向是：

1. **结构化重构** → 提升可维护性
2. **交互流程优化** → 减少操作步骤
3. **视觉层级强化** → 卡片化、模块化
4. **CSS 分离管理** → 提升可读性

重点是学习 Claude Code 的"任务卡片 + 上下文操作"模式，而不是简单模仿样式。
