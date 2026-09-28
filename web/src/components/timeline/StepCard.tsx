import { Check, ChevronRight, Loader2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { formatDuration } from "@/lib/format";
import type { StepChild, StepItem } from "@/store/timeline";
import { ModelCallRow } from "./ModelCallRow";
import { ThinkingBlock } from "./ThinkingBlock";
import { ToolRow } from "./ToolRow";

function ChildRow({ child }: { child: StepChild }) {
  if (child.kind === "model_call") return <ModelCallRow call={child} />;
  return <ToolRow tool={child} />;
}

function metadataSummary(metadata: Record<string, unknown> | undefined): string {
  if (!metadata) return "";
  const parts: string[] = [];
  for (const [key, value] of Object.entries(metadata)) {
    if (value == null) continue;
    parts.push(`${key}: ${typeof value === "object" ? JSON.stringify(value) : String(value)}`);
  }
  return parts.join(" · ");
}

// 步骤卡片（计划 §9.2）：默认折叠；运行中自动展开、完成后自动折叠，
// 用户手动展开/收起过后不再被自动行为覆盖。
export function StepCard({ step }: { step: StepItem }) {
  const [userToggled, setUserToggled] = useState<boolean | null>(null);
  const [now, setNow] = useState(() => Date.now() / 1000);

  useEffect(() => {
    if (step.status !== "running") return;
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, [step.status]);

  const expanded = userToggled ?? step.status === "running";
  const duration =
    step.status === "running" ? Math.max(0, now - step.startedAt) : step.duration ?? null;
  const summary = metadataSummary(step.metadata);

  return (
    <div
      className={cn(
        "rounded-xl border",
        step.status === "failed" && "border-red-300 dark:border-red-900/60",
      )}
    >
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
        {step.status === "running" ? (
          <Loader2 className="size-3.5 shrink-0 animate-spin text-blue-500" />
        ) : step.status === "failed" ? (
          <X className="size-3.5 shrink-0 text-red-500" />
        ) : (
          <Check className="size-3.5 shrink-0 text-muted-foreground/70" />
        )}
        <span className="min-w-0 flex-1 truncate">{step.label}</span>
        {duration != null && (
          <span className="shrink-0 font-mono text-xs text-muted-foreground">
            {formatDuration(duration)}
          </span>
        )}
      </button>
      {expanded && (step.thinking || step.children.length > 0 || summary) && (
        <div className="space-y-2 px-3 pb-3">
          <ThinkingBlock text={step.thinking} />
          {step.children.map((child, index) => (
            <ChildRow key={`${child.kind}:${child.id}:${index}`} child={child} />
          ))}
          {summary && (
            <p className="truncate font-mono text-[11px] text-muted-foreground/80" title={summary}>
              {summary}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
