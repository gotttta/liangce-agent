import { cn } from "@/lib/utils";

// 任务/运行状态的小圆点颜色；状态中文名在 lib/format.ts
const STATUS_DOT_CLASS: Record<string, string> = {
  draft: "bg-muted-foreground/40",
  in_progress: "bg-blue-500 animate-pulse",
  running: "bg-blue-500 animate-pulse",
  waiting_for_feedback: "bg-amber-500",
  awaiting_feedback: "bg-amber-500",
  waiting_for_acceptance: "bg-amber-500",
  completed: "bg-emerald-500",
  accepted: "bg-emerald-500",
  failed: "bg-red-500",
  stopped: "bg-muted-foreground/60",
  cancelled: "bg-muted-foreground/60",
  interrupted: "bg-muted-foreground/60",
  exited: "bg-muted-foreground/60",
};

export function statusDotClass(status: string, extra?: string): string {
  return cn(STATUS_DOT_CLASS[status] ?? "bg-muted-foreground/40", extra);
}
