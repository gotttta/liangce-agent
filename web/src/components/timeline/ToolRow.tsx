import { Check, Wrench, X } from "lucide-react";
import { formatDuration } from "@/lib/format";
import type { ToolChild } from "@/store/timeline";

function preview(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

// 工具调用行：参数 / 结果折叠显示
export function ToolRow({ tool }: { tool: ToolChild }) {
  const finished = tool.result !== undefined;
  return (
    <div className="rounded-lg border px-3 py-2 text-xs">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <Wrench className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="font-mono font-medium">{tool.tool}</span>
        {tool.success === false ? (
          <span className="flex items-center gap-1 text-red-500">
            <X className="size-3" /> 失败
          </span>
        ) : finished ? (
          <span className="flex items-center gap-1 text-emerald-600">
            <Check className="size-3" /> 成功
          </span>
        ) : (
          <span className="text-muted-foreground">执行中…</span>
        )}
        {tool.duration != null && (
          <span className="text-muted-foreground">{formatDuration(tool.duration)}</span>
        )}
      </div>
      <details className="mt-1.5">
        <summary className="cursor-pointer select-none text-muted-foreground hover:text-foreground">
          参数与结果
        </summary>
        <div className="mt-1 space-y-1.5">
          <div className="rounded bg-muted/50 p-2 font-mono text-[11px] leading-4 whitespace-pre-wrap break-all text-muted-foreground">
            参数：{preview(tool.args)}
          </div>
          {tool.result !== undefined && (
            <div className="rounded bg-muted/50 p-2 font-mono text-[11px] leading-4 whitespace-pre-wrap break-all text-muted-foreground">
              结果：{preview(tool.result)}
            </div>
          )}
        </div>
      </details>
    </div>
  );
}
