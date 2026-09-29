import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router";
import { api } from "@/api/client";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { AlgorithmView } from "./AlgorithmView";
import { ReferenceGallery } from "./ReferenceGallery";
import { ScoreChart } from "./ScoreChart";

type ArtifactTab = "reference" | "iteration" | "algorithm";

const TABS: { key: ArtifactTab; label: string }[] = [
  { key: "reference", label: "参考" },
  { key: "iteration", label: "迭代" },
  { key: "algorithm", label: "算法" },
];

// 产物面板（计划 §9.6）：参考 / 迭代 / 算法三个标签页。
// 数据来自 GET /artifacts；TaskPage 在收到 iteration_scored、reference_candidate、
// run_finished 事件时节流失效重取（queryKey 前缀 ["artifacts", taskId]）。
export function ArtifactPanel() {
  const { taskId } = useParams();
  const [tab, setTab] = useState<ArtifactTab>("reference");

  const task = useQuery({
    queryKey: ["task", taskId],
    queryFn: () => api.getTask(taskId!),
    enabled: Boolean(taskId),
  });
  // 产物指向最新 run：latest_run_id 优先，运行窗口内用 active_run_id，
  // 两者都缺时回退 runs 列表最后一条（runs 按开始时间升序）
  const detail = task.data;
  const fallbackRunId =
    detail && detail.runs.length > 0 ? detail.runs[detail.runs.length - 1].run_id : null;
  const runId = detail
    ? (detail.latest_run_id ?? detail.active_run_id ?? fallbackRunId)
    : null;
  const artifacts = useQuery({
    queryKey: ["artifacts", taskId, runId],
    queryFn: () => api.getArtifacts(taskId!, runId!),
    enabled: Boolean(taskId && runId),
  });

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 gap-1 border-b px-3 pt-2" role="tablist" aria-label="产物视图">
        {TABS.map((item) => (
          <button
            key={item.key}
            type="button"
            role="tab"
            aria-selected={tab === item.key}
            onClick={() => setTab(item.key)}
            className={cn(
              "rounded-t-lg px-3 py-1.5 text-sm transition-colors hover:bg-muted/50",
              tab === item.key
                ? "border border-b-transparent bg-muted/60 font-medium"
                : "text-muted-foreground",
            )}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {!taskId || !task.data ? (
          <PanelSkeleton />
        ) : !runId ? (
          <EmptyHint text="启动一次运行后，这里会展示参考掩膜、迭代分数与最优算法。" />
        ) : artifacts.isPending ? (
          <PanelSkeleton />
        ) : artifacts.isError ? (
          <EmptyHint
            text={`产物加载失败：${artifacts.error instanceof Error ? artifacts.error.message : "未知错误"}`}
          />
        ) : tab === "reference" ? (
          <ReferenceGallery
            samples={task.data.samples}
            references={artifacts.data?.references ?? []}
          />
        ) : tab === "iteration" ? (
          <ScoreChart iterations={artifacts.data?.iterations ?? []} />
        ) : (
          <AlgorithmView
            taskId={taskId}
            runId={runId}
            algorithm={artifacts.data?.algorithm ?? null}
            iterations={artifacts.data?.iterations ?? []}
          />
        )}
      </div>
    </div>
  );
}

export function EmptyHint({ text }: { text: string }) {
  return (
    <div className="flex h-full items-center justify-center p-6 text-center text-sm text-muted-foreground">
      {text}
    </div>
  );
}

function PanelSkeleton() {
  return (
    <div className="space-y-3">
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-24 w-3/4" />
    </div>
  );
}
