import { ArrowUp, CircleAlert, Plus, Square } from "lucide-react";
import { useCallback, useRef, useState, type ClipboardEvent, type KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { SampleOut } from "@/api/types";
import type { ReviewItem } from "@/store/timeline";
import { AttachmentChips } from "./AttachmentChips";

const SAMPLE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"];

export type TargetType = "defect" | "array";

interface ComposerProps {
  samples: SampleOut[];
  uploading: string[];
  hasRuns: boolean;
  running: boolean;
  pendingReview: ReviewItem | null;
  onStart: (message: string, targetType: TargetType) => Promise<void>;
  onReview: (feedback: string) => Promise<void>;
  onCancel: () => void;
  onRemoveSample: (name: string) => void;
  onAddFiles: (files: File[]) => void;
}

// 输入区（计划 §9.3）：Enter 发送、Shift+Enter 换行、输入法组字不发送；
// 运行中变“停止”；停在人工确认点时发文字等价于 continue + feedback。
export function Composer({
  samples,
  uploading,
  hasRuns,
  running,
  pendingReview,
  onStart,
  onReview,
  onCancel,
  onRemoveSample,
  onAddFiles,
}: ComposerProps) {
  const [text, setText] = useState("");
  const [targetType, setTargetType] = useState<TargetType>("defect");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const [dragOver, setDragOver] = useState(false);

  const canSend =
    !running && !sending && text.trim().length > 0 && (samples.length > 0 || pendingReview !== null);

  const send = useCallback(async () => {
    if (!canSend) {
      if (running) return;
      if (samples.length === 0 && pendingReview === null) {
        setError("请先添加样本图");
        return;
      }
      return;
    }
    const message = text.trim();
    setSending(true);
    setError(null);
    try {
      if (pendingReview) {
        await onReview(message);
      } else {
        await onStart(message, targetType);
      }
      setText("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "发送失败，请重试");
    } finally {
      setSending(false);
      textareaRef.current?.focus();
    }
  }, [canSend, running, samples.length, pendingReview, text, onStart, onReview, targetType]);

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    if (running) return; // 运行中不能发送，Enter 让位于默认行为
    event.preventDefault();
    void send();
  };

  const onPaste = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const files = Array.from(event.clipboardData.files).filter((file) =>
      SAMPLE_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext)),
    );
    if (files.length === 0) return;
    event.preventDefault();
    onAddFiles(files);
  };

  const placeholder = running
    ? "任务运行中，可先输入内容，停止后发送…"
    : pendingReview
      ? "输入修正意见，或使用上方按钮确认"
      : "描述要检测的缺陷（Enter 发送，Shift+Enter 换行，可直接粘贴或拖入样本图）";

  return (
    <div
      className={cn(
        "border-t bg-background px-6 py-4 transition-colors",
        dragOver && "bg-primary/5",
      )}
      onDragEnter={(event) => {
        if (!event.dataTransfer.types.includes("Files")) return;
        dragDepth.current += 1;
        setDragOver(true);
      }}
      onDragOver={(event) => {
        if (event.dataTransfer.types.includes("Files")) event.preventDefault();
      }}
      onDragLeave={() => {
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (dragDepth.current === 0) setDragOver(false);
      }}
      onDrop={(event) => {
        if (!event.dataTransfer.types.includes("Files")) return;
        event.preventDefault();
        dragDepth.current = 0;
        setDragOver(false);
        const files = Array.from(event.dataTransfer.files).filter((file) =>
          SAMPLE_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext)),
        );
        if (files.length > 0) onAddFiles(files);
      }}
    >
      <div className="mx-auto max-w-3xl">
        {!hasRuns && !running && (
          <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
            <span>目标类型</span>
            <div className="flex overflow-hidden rounded-lg border" role="group" aria-label="目标类型">
              <button
                type="button"
                onClick={() => setTargetType("defect")}
                aria-pressed={targetType === "defect"}
                className={cn(
                  "px-3 py-1 transition-colors hover:bg-muted",
                  targetType === "defect" && "bg-muted font-medium text-foreground",
                )}
              >
                缺陷
              </button>
              <button
                type="button"
                onClick={() => setTargetType("array")}
                aria-pressed={targetType === "array"}
                className={cn(
                  "border-l px-3 py-1 transition-colors hover:bg-muted",
                  targetType === "array" && "bg-muted font-medium text-foreground",
                )}
              >
                阵列
              </button>
            </div>
          </div>
        )}

        <AttachmentChips samples={samples} uploading={uploading} onRemove={onRemoveSample} />

        <div className="flex items-end gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept={SAMPLE_EXTENSIONS.join(",")}
            multiple
            className="hidden"
            onChange={(event) => {
              const files = Array.from(event.target.files ?? []);
              if (files.length > 0) onAddFiles(files);
              event.target.value = "";
            }}
          />
          <Button
            variant="outline"
            size="icon"
            className="size-9 shrink-0"
            onClick={() => fileInputRef.current?.click()}
            aria-label="添加样本图"
            title="添加样本图"
          >
            <Plus className="size-4" />
          </Button>
          <Textarea
            ref={textareaRef}
            value={text}
            placeholder={placeholder}
            rows={1}
            // field-sizing-content 自动增高，max-h 封顶 8 行后内部滚动
            className="max-h-48 min-h-9 flex-1 resize-none overflow-y-auto rounded-xl px-3 py-2 text-sm"
            aria-label="任务描述"
            onChange={(event) => setText(event.target.value)}
            onKeyDown={onKeyDown}
            onPaste={onPaste}
          />
          {running ? (
            <Button
              size="icon"
              variant="outline"
              className="size-9 shrink-0"
              onClick={onCancel}
              aria-label="停止运行"
              title="停止运行"
            >
              <Square className="size-3.5 fill-current" />
            </Button>
          ) : (
            <Button
              size="icon"
              className="size-9 shrink-0"
              disabled={!canSend}
              onClick={() => void send()}
              aria-label={pendingReview ? "发送修正意见" : "发送"}
              title={pendingReview ? "发送修正意见" : "发送"}
            >
              <ArrowUp className="size-4" />
            </Button>
          )}
        </div>

        {samples.length === 0 && !pendingReview && !running && (
          <p className="mt-2 px-1 text-xs text-muted-foreground">请先添加样本图</p>
        )}
        {error && (
          <p className="mt-2 flex items-center gap-1.5 px-1 text-xs text-red-600" role="alert">
            <CircleAlert className="size-3.5" />
            {error}
          </p>
        )}
      </div>
    </div>
  );
}
