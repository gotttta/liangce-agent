import { CircleAlert, CircleHelp, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { ApiError } from "@/api/client";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { toast } from "@/components/ui/toast";
import { Textarea } from "@/components/ui/textarea";
import { formatPipeline, formatScore, stopReasonLabel } from "@/lib/format";
import type { ReviewItem, ReviewResolved } from "@/store/timeline";
import { ImageViewer } from "./ImageViewer";

export type ReviewAction = "accept" | "continue" | "exit";

interface ReviewCardProps {
  review: ReviewItem;
  onSubmit: (action: ReviewAction, feedback?: string) => Promise<void>;
  /** final 阶段展示最优算法摘要（来自 TaskDetail.best） */
  bestPipeline?: Record<string, unknown>[] | null;
  /** “在右侧查看详情”入口（切换产物面板） */
  onOpenArtifacts?: () => void;
}

function resolvedLabel(resolved: ReviewResolved): string {
  switch (resolved.action) {
    case "continue":
      return resolved.feedback ? `已提交修正：${resolved.feedback}` : "已确认";
    case "accept":
      return "已接受结果";
    case "exit":
      return "已结束任务";
    default:
      return "已处理";
  }
}

// 人工确认卡片（计划 §9.4）：reference / final / retry 三种形态，
// 提交后按钮立即禁用，防止重复请求。
export function ReviewCard({ review, onSubmit, bestPipeline, onOpenArtifacts }: ReviewCardProps) {
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [confirmExit, setConfirmExit] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [localResolved, setLocalResolved] = useState<ReviewResolved | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewerOpen, setViewerOpen] = useState(false);

  const resolved = review.resolved ?? localResolved;
  const disabled = resolved !== null || submitting;

  const submit = async (action: ReviewAction, feedbackText?: string) => {
    setSubmitting(true);
    setError(null);
    try {
      await onSubmit(action, feedbackText);
      setLocalResolved(
        feedbackText ? { action, feedback: feedbackText } : { action },
      );
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 409) {
        // 任务仍在运行（如另一标签页刚提交过）：toast 提示并保持可重试
        toast("任务仍在运行，请等待本次运行结束后再操作。", "error");
      } else {
        setError(cause instanceof Error ? cause.message : "提交失败，请重试");
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="rounded-xl border border-amber-300/70 bg-amber-50/40 p-4 dark:border-amber-800/60 dark:bg-amber-950/20">
      <div className="flex items-center gap-2 text-sm font-medium">
        <CircleHelp className="size-4 text-amber-600 dark:text-amber-500" />
        {review.stage === "reference"
          ? "请确认参考掩膜"
          : review.stage === "final"
            ? "迭代已停止，请确认结果"
            : "需要重试"}
      </div>

      {review.stage === "reference" && review.overlayUrl && (
        <button
          type="button"
          onClick={() => setViewerOpen(true)}
          className="mt-3 block overflow-hidden rounded-lg border bg-background hover:opacity-90"
          aria-label="查看参考掩膜叠加图大图"
        >
          <img
            src={review.overlayUrl}
            alt="参考掩膜叠加图"
            className="max-h-72 w-auto object-contain"
          />
        </button>
      )}
      {review.stage === "reference" && review.overlayUrl && (
        <ImageViewer
          src={review.overlayUrl}
          alt="参考掩膜叠加图"
          open={viewerOpen}
          onOpenChange={setViewerOpen}
        />
      )}

      {review.message && (
        <p className="mt-3 text-sm leading-relaxed text-foreground/90">{review.message}</p>
      )}

      {review.stage === "final" && (
        <div className="mt-3 space-y-1.5 rounded-lg bg-background/70 p-3 text-sm">
          {review.stopReason && (
            <p className="flex items-center gap-2">
              <ShieldCheck className="size-4 text-emerald-600" />
              停止原因：{stopReasonLabel(review.stopReason)}
            </p>
          )}
          <p>
            最优分数 <span className="font-mono font-medium">{formatScore(review.bestScore)}</span>
          </p>
          {bestPipeline && bestPipeline.length > 0 && (
            <p className="truncate font-mono text-xs text-muted-foreground" title={formatPipeline(bestPipeline)}>
              {formatPipeline(bestPipeline)}
            </p>
          )}
          {onOpenArtifacts && (
            <button
              type="button"
              onClick={onOpenArtifacts}
              className="text-xs text-primary underline-offset-4 hover:underline"
            >
              在右侧查看详情
            </button>
          )}
        </div>
      )}

      {resolved ? (
        <p className="mt-3 text-xs text-muted-foreground">✓ {resolvedLabel(resolved)}</p>
      ) : (
        <div className="mt-3 space-y-2">
          <div className="flex flex-wrap gap-2">
            {review.stage === "reference" && (
              <>
                <Button size="sm" disabled={disabled} onClick={() => submit("continue")}>
                  确认
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={disabled}
                  onClick={() => {
                    setFeedbackOpen(!feedbackOpen);
                  }}
                >
                  需要修正
                </Button>
              </>
            )}
            {review.stage === "final" && (
              <>
                <Button size="sm" disabled={disabled} onClick={() => submit("accept")}>
                  接受结果
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={disabled}
                  onClick={() => setFeedbackOpen(!feedbackOpen)}
                >
                  继续迭代
                </Button>
              </>
            )}
            {review.stage === "retry" && (
              <Button size="sm" disabled={disabled} onClick={() => submit("continue")}>
                重试
              </Button>
            )}
            <Button
              size="sm"
              variant="ghost"
              className="text-red-600 hover:bg-red-50 dark:hover:bg-red-950/40"
              disabled={disabled}
              onClick={() => setConfirmExit(true)}
            >
              放弃任务
            </Button>
          </div>

          {feedbackOpen && (
            <div className="space-y-2">
              <Textarea
                value={feedback}
                onChange={(event) => setFeedback(event.target.value)}
                placeholder={
                  review.stage === "reference"
                    ? "描述掩膜哪里不对或遗漏（例如：边缘偏小，请包含完整的亮块）"
                    : "补充迭代方向（可选）"
                }
                className="min-h-16 text-sm"
                aria-label="修正意见"
              />
              <Button
                size="sm"
                disabled={disabled || !feedback.trim()}
                onClick={() => submit("continue", feedback.trim())}
              >
                提交{review.stage === "reference" ? "修正" : "并继续迭代"}
              </Button>
            </div>
          )}

          {error && (
            <p className="flex items-center gap-1.5 text-xs text-red-600">
              <CircleAlert className="size-3.5" />
              {error}
            </p>
          )}
        </div>
      )}

      <Dialog open={confirmExit} onOpenChange={setConfirmExit}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>放弃任务</DialogTitle>
            <DialogDescription>
              确认放弃？本任务的参考掩膜与迭代记录会保留，但不会再继续优化。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmExit(false)}>
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                setConfirmExit(false);
                void submit("exit");
              }}
            >
              放弃任务
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
