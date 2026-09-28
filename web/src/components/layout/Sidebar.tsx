import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Ellipsis, Pencil, Plus, Trash2 } from "lucide-react";
import { useRef, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { api } from "@/api/client";
import type { TaskSummary } from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { statusDotClass } from "@/lib/status";
import { cn } from "@/lib/utils";

function TaskRow({ task }: { task: TaskSummary }) {
  const { taskId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(task.title);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const active = taskId === task.id;

  const rename = useMutation({
    mutationFn: (title: string) => api.patchTask(task.id, title),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tasks"] });
      queryClient.invalidateQueries({ queryKey: ["task", task.id] });
    },
  });
  const remove = useMutation({
    mutationFn: () => api.deleteTask(task.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tasks"] });
      if (active) navigate("/");
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
    <div
      className={cn(
        "group relative flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm",
        active ? "bg-muted" : "hover:bg-muted/60",
      )}
    >
      <button
        type="button"
        onClick={() => navigate(`/tasks/${task.id}`)}
        className="flex min-w-0 flex-1 items-center gap-2 text-left"
      >
        <span
          className={cn("size-2 shrink-0 rounded-full", statusDotClass(task.status))}
          aria-label={task.status_label}
        />
        {editing ? (
          <input
            ref={inputRef}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={commit}
            onClick={(event) => event.stopPropagation()}
            onKeyDown={(event) => {
              if (event.key === "Enter") commit();
              if (event.key === "Escape") setEditing(false);
            }}
            className="h-7 w-full rounded-md border bg-transparent px-1.5 text-sm outline-none focus:ring-1 focus:ring-ring"
            aria-label="任务标题"
          />
        ) : (
          <span className="truncate">{task.title}</span>
        )}
      </button>
      {task.running ? (
        <span className="shrink-0 text-[10px] text-blue-500">运行中</span>
      ) : null}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="size-6 shrink-0 opacity-0 group-hover:opacity-100 data-[state=open]:opacity-100"
            aria-label="任务操作"
          >
            <Ellipsis className="size-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={startEdit}>
            <Pencil className="size-4" /> 重命名
          </DropdownMenuItem>
          <DropdownMenuItem variant="destructive" onSelect={() => setConfirmDelete(true)}>
            <Trash2 className="size-4" /> 删除
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>删除任务</DialogTitle>
            <DialogDescription>
              确认删除「{task.title}」？任务的样本图、参考掩膜和运行记录会一并删除，无法恢复。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmDelete(false)}>
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                setConfirmDelete(false);
                remove.mutate();
              }}
            >
              删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export function Sidebar() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const tasks = useQuery({
    queryKey: ["tasks"],
    queryFn: () => api.listTasks(),
    refetchInterval: 5_000,
  });
  const create = useMutation({
    mutationFn: () => api.createTask(),
    onSuccess: (task) => {
      queryClient.invalidateQueries({ queryKey: ["tasks"] });
      navigate(`/tasks/${task.id}`);
    },
  });

  return (
    <aside className="flex h-full min-h-0 flex-col border-r bg-muted/20">
      <div className="flex items-center justify-between px-3 py-3">
        <span className="text-sm font-semibold">任务</span>
        <Button
          variant="outline"
          size="sm"
          onClick={() => create.mutate()}
          disabled={create.isPending}
          aria-label="新建任务"
        >
          <Plus className="size-4" /> 新建
        </Button>
      </div>
      <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-2">
        {tasks.isLoading ? (
          <div className="px-2 py-2 text-xs text-muted-foreground">加载中…</div>
        ) : tasks.data ? (
          tasks.data.map((task) => <TaskRow key={task.id} task={task} />)
        ) : null}
        {tasks.data?.length === 0 ? (
          <div className="px-2 py-2 text-xs text-muted-foreground">
            还没有任务，点击「新建」开始
          </div>
        ) : null}
      </nav>
      <div className="border-t px-3 py-2 text-[11px] text-muted-foreground">本地工作区</div>
    </aside>
  );
}
