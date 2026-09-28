import { Sparkles } from "lucide-react";
import { formatDuration, formatTokens } from "@/lib/format";
import type { ModelCallChild } from "@/store/timeline";

// 模型调用行：模型名 · token 用量 · 耗时；输出文本折叠显示（计划 §9.2）
export function ModelCallRow({ call }: { call: ModelCallChild }) {
  return (
    <div className="rounded-lg border px-3 py-2 text-xs">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <Sparkles className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="font-medium">{call.model || "模型调用"}</span>
        {call.status === "running" ? (
          <span className="text-muted-foreground">请求中…</span>
        ) : (
          <>
            {formatTokens(call.usage) && (
              <span className="text-muted-foreground">{formatTokens(call.usage)}</span>
            )}
            {call.duration != null && (
              <span className="text-muted-foreground">{formatDuration(call.duration)}</span>
            )}
          </>
        )}
      </div>
      {call.output ? (
        <details className="mt-1.5">
          <summary className="cursor-pointer select-none text-muted-foreground hover:text-foreground">
            输出内容
          </summary>
          <div className="mt-1 max-h-40 overflow-y-auto rounded bg-muted/50 p-2 font-mono text-[11px] leading-4 whitespace-pre-wrap break-words text-muted-foreground">
            {call.output}
          </div>
        </details>
      ) : null}
    </div>
  );
}
