import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { PanelRight } from "lucide-react";
import { useRef, useState } from "react";
import { api } from "@/api/client";
import type { TaskSummary } from "@/api/types";
import { Button } from "@/components/ui/button";
import { statusDotClass } from "@/lib/status";
import { useUiStore } from "@/store/ui";

export function TopBar({ task }: { task: TaskSummary }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(task.title);
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();
  const toggleRightPanel = useUiStore((state) => state.toggleRightPanel);
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
        {health.data?.model ? <span>{health.data.model}</span> : null}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleRightPanel}
          aria-label="切换产物面板"
          className="size-8"
        >
          <PanelRight className="size-4" />
        </Button>
      </div>
    </header>
  );
}
