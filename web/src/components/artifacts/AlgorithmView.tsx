import { Braces, Copy, Download, Layers } from "lucide-react";
import { useState } from "react";
import type { Artifacts, IterationRecord } from "@/api/types";
import { Button } from "@/components/ui/button";
import { EmptyHint } from "./ArtifactPanel";

// 算法标签页（计划 §9.6）：最优算子序列按步骤卡片显示（op 名 + 参数表），
// 可切换原始 JSON（带复制）；algorithm.json 落盘后可下载。
// finish 前没有 algorithm.json 时，回退显示最优一轮的算法草稿。

interface AlgorithmViewProps {
  taskId: string;
  runId: string;
  algorithm: Artifacts["algorithm"];
  iterations: IterationRecord[];
}

function paramEntries(step: Record<string, unknown>): [string, unknown][] {
  return Object.entries(step).filter(([key]) => key !== "op");
}

function algorithmSource(
  algorithm: Artifacts["algorithm"],
  iterations: IterationRecord[],
): { pipeline: Record<string, unknown>[]; notes: string; from: "final" | "draft" } | null {
  if (algorithm?.pipeline && algorithm.pipeline.length > 0) {
    return { pipeline: algorithm.pipeline, notes: algorithm.notes ?? "", from: "final" };
  }
  if (iterations.length === 0) return null;
  const best = iterations.reduce((acc, record) =>
    record.run_score.composite_mean > acc.run_score.composite_mean ? record : acc,
  );
  const pipeline = best.algorithm_spec?.pipeline ?? [];
  if (pipeline.length === 0) return null;
  return { pipeline, notes: best.algorithm_spec?.notes ?? "", from: "draft" };
}

export function AlgorithmView({ taskId, runId, algorithm, iterations }: AlgorithmViewProps) {
  const [showJson, setShowJson] = useState(false);
  const [copied, setCopied] = useState(false);
  const source = algorithmSource(algorithm, iterations);

  if (!source) {
    return <EmptyHint text="还没有最优算法；迭代出结果后这里会显示算子序列。" />;
  }

  const downloadUrl = `/api/tasks/${encodeURIComponent(taskId)}/runs/${encodeURIComponent(runId)}/algorithm.json`;

  const copyJson = async () => {
    try {
      await navigator.clipboard.writeText(
        JSON.stringify({ pipeline: source.pipeline, notes: source.notes }, null, 2),
      );
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // 剪贴板不可用（如非安全上下文）：按钮反馈退化为无操作
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="flex items-center gap-1.5 text-sm font-medium">
          <Layers className="size-4 text-muted-foreground" />
          最优算法
        </span>
        {source.from === "draft" && (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] text-amber-700 dark:bg-amber-950/60 dark:text-amber-400">
            迭代中的最优草稿
          </span>
        )}
        <span className="ml-auto flex items-center gap-1.5">
          <Button
            variant="ghost"
            size="xs"
            onClick={() => setShowJson(!showJson)}
            aria-pressed={showJson}
            aria-label="切换原始 JSON 视图"
          >
            <Braces className="size-3.5" /> JSON
          </Button>
          {algorithm && (
            <a
              href={downloadUrl}
              download
              className="inline-flex h-6 items-center gap-1 rounded-[min(var(--radius-md),10px)] border border-border bg-background px-2 text-xs font-medium hover:bg-muted"
              aria-label="下载 algorithm.json"
            >
              <Download className="size-3.5" /> 下载
            </a>
          )}
        </span>
      </div>

      {showJson ? (
        <div className="relative">
          <Button
            variant="ghost"
            size="icon-xs"
            className="absolute top-2 right-2"
            onClick={() => void copyJson()}
            aria-label="复制 JSON"
          >
            <Copy className="size-3.5" />
          </Button>
          {copied && (
            <span className="absolute top-3 right-10 text-[11px] text-emerald-600">已复制</span>
          )}
          <pre className="overflow-x-auto rounded-lg bg-muted/50 p-3 font-mono text-[11px] leading-4 text-muted-foreground">
            {JSON.stringify({ pipeline: source.pipeline, notes: source.notes }, null, 2)}
          </pre>
        </div>
      ) : (
        <div className="space-y-2">
          {source.pipeline.map((step, index) => {
            const params = paramEntries(step);
            return (
              <div key={index} className="rounded-xl border p-3">
                <div className="flex items-center gap-2 text-sm">
                  <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-[11px] font-medium text-muted-foreground">
                    {index + 1}
                  </span>
                  <span className="font-mono font-medium">{String(step.op ?? "?")}</span>
                </div>
                {params.length > 0 && (
                  <table className="mt-2 ml-7 text-left text-xs">
                    <tbody>
                      {params.map(([key, value]) => (
                        <tr key={key}>
                          <td className="pr-3 font-mono text-muted-foreground">{key}</td>
                          <td className="font-mono">{JSON.stringify(value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            );
          })}
          {source.notes && (
            <p className="px-1 text-xs text-muted-foreground">说明：{source.notes}</p>
          )}
        </div>
      )}
    </div>
  );
}
