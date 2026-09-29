import { useEffect, useRef } from "react";
import { statusDotClass } from "@/lib/status";
import { formatDuration, runStatusLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TimelineEntry, TimelineItem } from "@/store/timeline";
import { AssistantMessage } from "./AssistantMessage";
import { ErrorCard } from "./ErrorCard";
import { IterationGroup } from "./IterationGroup";
import { ReferenceCard } from "./ReferenceCard";
import { ReviewCard, type ReviewAction } from "./ReviewCard";
import { StepCard } from "./StepCard";
import { UserMessage } from "./UserMessage";

function RunEndMark({ item }: { item: Extract<TimelineItem, { kind: "run_end" }> }) {
  return (
    <div className="flex items-center justify-center gap-2 py-1 text-xs text-muted-foreground">
      <span className={cn("size-1.5 rounded-full", statusDotClass(item.status))} />
      运行{runStatusLabel(item.status)}
      {item.duration != null && <span>· {formatDuration(item.duration)}</span>}
    </div>
  );
}

interface TimelineProps {
  entries: TimelineEntry[];
  onReviewSubmit: (runId: string, action: ReviewAction, feedback?: string) => Promise<void>;
  onRetry: (message: string) => void;
  onOpenArtifacts: () => void;
  bestPipeline?: Record<string, unknown>[] | null;
  /** 用于计算初始消息（重试按钮需要原始任务描述） */
  messageByRun: Record<string, string>;
}

// 时间线（计划 §9.1/§9.2）：居中 max-w-3xl，条目间留白；
// 新内容到达时若用户停在底部附近则自动跟随滚动。
export function Timeline({
  entries,
  onReviewSubmit,
  onRetry,
  onOpenArtifacts,
  bestPipeline,
  messageByRun,
}: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  // 内容签名：条目增减或运行中步骤变化时触发滚动判断
  const signature = entries
    .map((entry) => (entry.kind === "item" ? entry.item.id : `${entry.id}#${entry.items.length}`))
    .join("|");

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const nearBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 120;
    if (nearBottom) element.scrollTo({ top: element.scrollHeight });
  }, [signature]);

  // 初次挂载（历史回放完成）直接定位到底部
  useEffect(() => {
    const element = containerRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, []);

  return (
    <div ref={containerRef} className="h-full overflow-y-auto" data-testid="timeline">
      <div className="mx-auto max-w-3xl space-y-3 px-6 py-8">
        {entries.length === 0 ? (
          <div className="flex min-h-48 flex-col items-center justify-center gap-2 rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">
            <span>还没有对话</span>
            <span className="text-xs">上传样本图、描述检测目标后开始；运行中的进度会实时显示在这里</span>
          </div>
        ) : null}
        {entries.map((entry) => {
          if (entry.kind === "iteration_group") {
            return <IterationGroup key={entry.id} group={entry} />;
          }
          const item = entry.item;
          switch (item.kind) {
            case "user":
              return <UserMessage key={item.id} item={item} />;
            case "assistant":
              return <AssistantMessage key={item.id} item={item} />;
            case "step":
              return <StepCard key={item.id} step={item} />;
            case "reference":
              return <ReferenceCard key={item.id} reference={item} />;
            case "iteration":
              return null; // 迭代条目全部在分组内渲染
            case "review":
              return (
                <ReviewCard
                  key={item.id}
                  review={item}
                  bestPipeline={bestPipeline}
                  onOpenArtifacts={onOpenArtifacts}
                  onSubmit={(action, feedback) =>
                    onReviewSubmit(item.runId, action, feedback)
                  }
                />
              );
            case "error":
              return (
                <ErrorCard
                  key={item.id}
                  error={item}
                  originalMessage={messageByRun[item.runId]}
                  onRetry={onRetry}
                />
              );
            case "run_end":
              return <RunEndMark key={item.id} item={item} />;
            default:
              return null;
          }
        })}
      </div>
    </div>
  );
}
