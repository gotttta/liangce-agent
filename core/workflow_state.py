from dataclasses import dataclass, field, fields
from typing import Annotated

from core.scoring import ImageScore, RunScore


@dataclass
class AlgorithmSpec:
    pipeline: list[dict]   # 算子序列，每个 dict 有 "op" 和参数键值对
    notes: str = ""


@dataclass
class WorkflowState:
    # 输入
    image_paths: list[str] = field(default_factory=list)
    task: str = ""
    target_type: str = "defect"       # "defect" | "array"

    # 运行环境
    run_id: str = ""
    output_root: str = ""
    run_dir: str = ""
    task_id: str = ""
    memory_context: dict = field(default_factory=dict)

    # 参考掩膜
    reference_confirmed: bool = False  # 用户已确认参考掩膜
    pending_reference_image_id: str = ""  # 等待用户确认的那张图
    pending_reference_mask_path: str = ""  # 候选掩膜落盘路径（等确认后写入 ReferenceStore）
    pending_reference_overlay_path: str = ""  # 叠加图路径，供确认界面展示
    pending_reference_score: float = 0.0  # 候选掩膜的 SAM IoU 分数均值

    # 迭代状态
    current_spec: AlgorithmSpec | None = None
    best_spec: AlgorithmSpec | None = None
    best_score: float = 0.0
    best_run_score: RunScore | None = None
    last_run_score: RunScore | None = None
    iteration: int = 0

    # 路由
    next_node: str = ""               # 由各节点设置，route() 读取
    stop_reason: str = ""             # "target_reached" | "no_improvement" | "max_iterations" | ""
    grace_iterations: int = 0         # 用户要求继续后跳过停止检查的轮数

    # 用户交互
    human_message: str = ""           # 展示给用户的消息
    user_feedback: str = ""           # 用户上一轮的反馈文字


def _replace(previous: WorkflowState, current: WorkflowState) -> WorkflowState:
    """全量替换，不合并；避免 LangGraph 对字段做默认追加。"""
    return current


WorkflowStateChannel = Annotated[WorkflowState, _replace]


def _restore_run_score(data: dict) -> RunScore:
    return RunScore(
        image_scores=[ImageScore(**item) for item in data.get("image_scores", [])],
        composite_mean=data.get("composite_mean", 0.0),
    )


def restore_state(state: "WorkflowState | dict") -> WorkflowState:
    """checkpoint 反序列化可能把 dataclass 退化成 dict；这里无损恢复成 WorkflowState。"""
    if isinstance(state, WorkflowState):
        return state
    if not isinstance(state, dict):
        raise TypeError(f"unexpected workflow state: {type(state)!r}")
    known = {field.name for field in fields(WorkflowState)}
    data = {key: value for key, value in state.items() if key in known}
    for key in ("current_spec", "best_spec"):
        if isinstance(data.get(key), dict):
            data[key] = AlgorithmSpec(**data[key])
    for key in ("best_run_score", "last_run_score"):
        if isinstance(data.get(key), dict):
            data[key] = _restore_run_score(data[key])
    return WorkflowState(**data)
