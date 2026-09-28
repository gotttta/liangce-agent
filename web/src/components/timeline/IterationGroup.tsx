import { ChevronRight, Loader2, TrendingUp } from "lucide-react";
import { useState } from "react";
import { formatPipeline, formatScore } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { IterationGroupEntry, TimelineItem } from "@/store/timeline";
import { StepCard } from "./StepCard";

function IterationRow({ item }: { item: Extract<TimelineItem, { kind: "iteration" }> }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="rounded-lg border bg-background">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 rounded-lg px-3 py-2 text-left text-xs hover:bg-muted/40"
      >
        <ChevronRight
          className={cn(
            "size-3 shrink-0 text-muted-foreground transition-transform",
            expanded && "rotate-90",
          )}
        />
        <span className="font-medium">第 {item.iteration} 轮</span>
        <span className="font-mono">{formatScore(item.score)}</span>
        {item.improved ? (
          <span className="text-emerald-600">↑ 刷新最优</span>
        ) : (
          <span className="text-muted-foreground">— 未提升</span>
        )}
        <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-muted-foreground">
          {formatPipeline(item.pipeline)}
        </span>
      </button>
      {expanded && (
        <div className="space-y-2 px-3 pb-3 text-xs">
          {item.notes && <p className="text-muted-foreground">{item.notes}</p>}
          <div className="overflow-x-auto rounded bg-muted/50 p-2 font-mono text-[11px] whitespace-pre-wrap break-all text-muted-foreground">
            {JSON.stringify(item.pipeline, null, 2)}
          </div>
          {item.imageScores.length > 0 && (
            <table className="w-full text-left text-[11px]">
              <thead className="text-muted-foreground">
                <tr>
                  <th className="py-0.5 pr-3 font-normal">图片</th>
                  <th className="py-0.5 pr-3 font-normal">IoU</th>
                  <th className="py-0.5 pr-3 font-normal">误检</th>
                  <th className="py-0.5 pr-3 font-normal">漏检</th>
                  <th className="py-0.5 pr-3 font-normal">参考数</th>
                  <th className="py-0.5 font-normal">综合</th>
                </tr>
              </thead>
              <tbody>
                {item.imageScores.map((score) => (
                  <tr key={score.image_id} className="border-t">
                    <td className="py-0.5 pr-3">{score.image_id}</td>
                    <td className="py-0.5 pr-3 font-mono">{score.iou_mean.toFixed(3)}</td>
                    <td className="py-0.5 pr-3 font-mono">{score.false_positive_count}</td>
                    <td className="py-0.5 pr-3 font-mono">{score.false_negative_count}</td>
                    <td className="py-0.5 pr-3 font-mono">{score.ref_count}</td>
                    <td className="py-0.5 font-mono">{score.composite.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}

// 迭代分组（计划 §9.2）：连续的迭代轮默认折叠为一组，避免几十轮刷屏；
// 组内还有正在运行的步骤时自动展开，便于观察当前轮进展。
export function IterationGroup({ group }: { group: IterationGroupEntry }) {
  const [userToggled, setUserToggled] = useState<boolean | null>(null);
  const containsRunning = group.items.some(
    (item) => item.kind === "step" && item.status === "running",
  );
  const expanded = userToggled ?? containsRunning;

  return (
    <div className="rounded-xl border bg-muted/20">
      <button
        type="button"
        onClick={() => setUserToggled(!expanded)}
        aria-expanded={expanded}
        className="flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left text-sm hover:bg-muted/40"
      >
        <ChevronRight
          className={cn(
            "size-3.5 shrink-0 text-muted-foreground transition-transform",
            expanded && "rotate-90",
          )}
        />
        {containsRunning && (
          <Loader2 className="size-3.5 shrink-0 animate-spin text-blue-500" />
        )}
        <TrendingUp className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="flex-1 font-medium">
          迭代 {group.from === group.to ? group.from : `${group.from}–${group.to}`}
        </span>
        <span className="shrink-0 text-xs text-muted-foreground">
          最优 <span className="font-mono">{formatScore(group.best)}</span>
        </span>
      </button>
      {expanded && (
        <div className="space-y-2 px-3 pb-3">
          {group.items.map((item, index) =>
            item.kind === "iteration" ? (
              <IterationRow key={item.id} item={item} />
            ) : item.kind === "step" ? (
              <StepCard key={`${item.id}:${index}`} step={item} />
            ) : null,
          )}
        </div>
      )}
    </div>
  );
}
