// 时长、时间、分数格式化与状态中文名（与后端 TASK_STATUS_LABELS 一致）
export const TASK_STATUS_LABELS: Record<string, string> = {
  draft: "草稿",
  in_progress: "进行中",
  waiting_for_acceptance: "待确认",
  waiting_for_feedback: "待反馈",
  running: "进行中",
  awaiting_feedback: "待反馈",
  completed: "已完成",
  stopped: "已停止",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
  accepted: "已验收",
  exited: "已结束",
};

export function statusLabel(status: string): string {
  return TASK_STATUS_LABELS[status] ?? status;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(Math.floor(seconds % 60)).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

export function formatClock(tsSeconds: number): string {
  const date = new Date(tsSeconds * 1000);
  return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

export function formatTimestamp(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const sameYear = date.getFullYear() === new Date().getFullYear();
  const datePart = sameYear
    ? `${date.getMonth() + 1}月${date.getDate()}日`
    : `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;
  return `${datePart} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

export function formatScore(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(3);
}

// 停止原因中文说明（计划 §9.4）
export const STOP_REASON_LABELS: Record<string, string> = {
  target_reached: "达到目标分数",
  no_improvement: "连续多轮没有明显提升",
  max_iterations: "达到迭代上限",
  user_exited: "用户已结束",
};

export function stopReasonLabel(reason: string | null | undefined): string {
  if (!reason) return "";
  return STOP_REASON_LABELS[reason] ?? reason;
}

// 运行终态的中文名（run_finished.status，计划 §12.2：运行状态以此为准）
export const RUN_STATUS_LABELS: Record<string, string> = {
  completed: "已完成",
  awaiting_review: "待确认",
  failed: "失败",
  cancelled: "已取消",
};

export function runStatusLabel(status: string): string {
  return RUN_STATUS_LABELS[status] ?? status;
}

/** 算子序列一行摘要：normalize → clahe → adaptive_threshold */
export function formatPipeline(pipeline: Record<string, unknown>[]): string {
  return pipeline
    .map((step) => String(step.op ?? "?"))
    .join(" → ");
}

export function formatTokens(usage: Record<string, number> | null | undefined): string {
  const total = usage?.total_tokens;
  if (typeof total !== "number" || !Number.isFinite(total)) return "";
  return `${total.toLocaleString("zh-CN")} tokens`;
}
