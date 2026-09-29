import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Menu, PanelRight } from "lucide-react";
import { useRef, useState } from "react";
import { api } from "@/api/client";
import type { TaskSummary } from "@/api/types";
import { Button } from "@/components/ui/button";
import { useMediaQuery } from "@/lib/hooks";
import { statusDotClass } from "@/lib/status";
import { useTimelineStore } from "@/store/timeline";
import { useUiStore } from "@/store/ui";

// 上下文用量小圆环（计划 §9.7）：最近一次 model_call_finished 的
// prompt_tokens / context_window；口径沿用 ui/utils/formatters.py。
function ContextUsageRing() {
  const usage = useTimelineStore((state) => state.contextUsage);
  if (!usage || !usage.window || usage.window <= 0) return null;
  const percent = Math.max(0, Math.min(100, (usage.used / usage.window) * 100));
  const color =
    percent >= 90
      ? "var(--destructive)"
      : percent >= 70
        ? "#d97706"
        : "var(--muted-foreground)";
  const radius = 6.5;
  const circumference = 2 * Math.PI * radius;
  const title = `上次请求输入 ${Math.round(usage.used).toLocaleString("zh-CN")} / ${Math.round(usage.window).toLocaleString("zh-CN")} tokens（${percent.toFixed(0)}%）`;

  return (
    <span className="flex items-center gap-1 text-xs text-muted-foreground" title={title}>
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
        <circle cx="9" cy="9" r={radius} fill="none" stroke="var(--border)" strokeWidth="2.5" />
        <circle
          cx="9"
          cy="9"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="2.5"
          strokeDasharray={`${(percent / 100) * circumference} ${circumference}`}
          strokeDashoffset={0}
          transform="rotate(-90 9 9)"
          strokeLinecap="round"
        />
      </svg>
      {percent.toFixed(0)}%
    </span>
  );
}

export function TopBar({ task }: { task: TaskSummary }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(task.title);
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();
  const wide = useMediaQuery("(min-width: 1024px)");
  const toggleRightPanel = useUiStore((state) => state.toggleRightPanel);
  const setSidebarOpen = useUiStore((state) => state.setSidebarOpen);
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, staleTime: 60_000 });

  const rename = useMutation({
    mutationFn: (title: string) => api.patchTask(task.id, title),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tasks"] });
      queryClient.invalidateQueries({ queryKey: ["task", task.id] });
    },
  });

  const startEdit = () => {
    setDraft(task.title);
    setEditing(true);
    requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });
  };

  const commit = () => {
    setEditing(false);
    const title = draft.trim();
    if (title && title !== task.title) rename.mutate(title);
  };

  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b px-4">
      {!wide && (
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          onClick={() => setSidebarOpen(true)}
          aria-label="打开任务列表"
        >
          <Menu className="size-4" />
        </Button>
      )}
      {editing ? (
        <input
          ref={inputRef}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") commit();
            if (event.key === "Escape") setEditing(false);
          }}
          className="h-8 w-64 rounded-md border bg-transparent px-2 text-sm outline-none focus:ring-1 focus:ring-ring"
          aria-label="任务标题"
        />
      ) : (
        <button
          type="button"
          onClick={startEdit}
          className="max-w-80 truncate rounded-md px-1.5 py-1 text-sm font-medium hover:bg-muted"
          title="点击重命名"
        >
          {task.title}
        </button>
      )}
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <span className={statusDotClass(task.status, "size-2 rounded-full")} />
        {task.status_label}
      </span>
      <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
        <ContextUsageRing />
        {health.data?.model ? <span>{health.data.model}</span> : null}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleRightPanel}
          aria-label="切换产物面板"
          title="切换产物面板（Ctrl/Cmd+.）"
          className="size-8"
        >
          <PanelRight className="size-4" />
        </Button>
      </div>
    </header>
  );
}
