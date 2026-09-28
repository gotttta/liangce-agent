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
