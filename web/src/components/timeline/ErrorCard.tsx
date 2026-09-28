import { CircleAlert, RotateCcw } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import type { ErrorItem } from "@/store/timeline";

interface ErrorCardProps {
  error: ErrorItem;
  /** 重试 = 用同一 message 重新启动运行（计划 §9.2） */
  onRetry?: (message: string) => void;
  originalMessage?: string;
}

export function ErrorCard({ error, onRetry, originalMessage }: ErrorCardProps) {
  const [retrying, setRetrying] = useState(false);

  return (
    <div className="rounded-xl border border-red-300 bg-red-50/50 p-3 text-sm dark:border-red-900/60 dark:bg-red-950/20">
      <p className="flex items-start gap-2">
        <CircleAlert className="mt-0.5 size-4 shrink-0 text-red-500" />
        <span className="min-w-0 flex-1 whitespace-pre-wrap break-words text-red-700 dark:text-red-400">
          {error.message}
        </span>
      </p>
      {onRetry && originalMessage && (
        <div className="mt-2 flex justify-end">
          <Button
            size="sm"
            variant="outline"
            disabled={retrying}
            onClick={() => {
              setRetrying(true);
              onRetry(originalMessage);
            }}
          >
            <RotateCcw className="size-3.5" /> 重试
          </Button>
        </div>
      )}
    </div>
  );
}
