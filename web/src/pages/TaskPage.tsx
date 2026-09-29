import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router";
import { api } from "@/api/client";
import { subscribeRunEvents } from "@/api/sse";
import { Composer, type TargetType } from "@/components/composer/Composer";
import { Timeline } from "@/components/timeline/Timeline";
import type { ReviewAction } from "@/components/timeline/ReviewCard";
import { TopBar } from "@/components/layout/TopBar";
import { Skeleton } from "@/components/ui/skeleton";
import { useUiStore } from "@/store/ui";
import { groupTimeline, useTimelineStore } from "@/store/timeline";

// 收到这三类事件后产物面板的数据变了（计划 §9.6：节流 2 秒失效重取）
const ARTIFACT_EVENT_TYPES = new Set(["iteration_scored", "reference_candidate", "run_finished"]);

interface ManualRun {
  taskId: string;
  runId: string;
}

// 任务页数据流（计划 §9.8）：
//   任务详情 → 逐个回放历史 run 的持久化事件 → 重建时间线；
//   最新 run 在运行时订阅 SSE，断线由 EventSource 自动重连；
//   发送消息 / 提交决策后（重）开同一 run 的 SSE。
export function TaskPage() {
  const { taskId } = useParams();
  const queryClient = useQueryClient();
  const setRightPanel = useUiStore((state) => state.setRightPanel);

  const task = useQuery({
    queryKey: ["task", taskId],
    queryFn: () => api.getTask(taskId!),
    enabled: Boolean(taskId),
    refetchInterval: 5_000,
  });

  const items = useTimelineStore((state) => state.items);
  const runStatus = useTimelineStore((state) => state.runStatus);
  const beginTask = useTimelineStore((state) => state.beginTask);
  const pushEvent = useTimelineStore((state) => state.pushEvent);

  const loadedRuns = useRef<Set<string>>(new Set());
  // manual：本页发起的 start / review 接管的 run；server：任务详情报告的运行中 run
  const [manual, setManual] = useState<ManualRun | null>(null);
  const [sseEpoch, setSseEpoch] = useState(0);
  const [uploading, setUploading] = useState<string[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [replaying, setReplaying] = useState(false);
  // 初始回放完成的任务 ID：回放没完成前不开 SSE，否则运行中一轮的实时事件
  // 会插到旧运行前面，时间线顺序错乱（审查意见 #2）
  const [replayDoneFor, setReplayDoneFor] = useState<string | null>(null);
  const artifactsTimer = useRef<number | null>(null);

  // 切换任务时丢弃旧任务的 manual 接管（渲染期状态调整，避免 effect 里 setState）
  if (manual && manual.taskId !== taskId) setManual(null);

  const manualRunId = manual && manual.taskId === taskId ? manual.runId : null;
  // latest_run_id 要到第一次 human_gate 才落盘，运行窗口内以 active_run_id 为准
  const serverRunId = task.data?.running
    ? (task.data.active_run_id ?? task.data.latest_run_id)
    : null;
  const liveRunId = manualRunId ?? serverRunId;
  // 服务器“运行中”可能滞后于事件流：时间线里已有终态的 run 不算运行
  const running = manualRunId !== null || (serverRunId !== null && !(serverRunId in runStatus));

  // 产物数据节流失效（2 秒窗口合并连续事件）
  const scheduleArtifactsRefetch = useCallback(() => {
    if (artifactsTimer.current != null) return;
    artifactsTimer.current = window.setTimeout(() => {
      artifactsTimer.current = null;
      if (taskId) void queryClient.invalidateQueries({ queryKey: ["artifacts", taskId] });
    }, 2000);
  }, [queryClient, taskId]);

  useEffect(
    () => () => {
      if (artifactsTimer.current != null) window.clearTimeout(artifactsTimer.current);
    },
    [],
  );

  useEffect(() => {
    if (!taskId) return;
    loadedRuns.current = new Set();
    beginTask(taskId);
  }, [taskId, beginTask]);

  // 历史回放：对任务详情里出现的每个 run 拉快照重放（按顺序 await，保证 run 顺序）。
  // loadedRuns 只在事件推送成功后才标记：effect 被取消时未推送的 run 保持未标记，
  // 下次 task.data 变化会重新回放，不会出现“已标记但没推送”的丢数据（审查意见 #2）。
  // 当前活动 run 跳过快照回放，其历史由 SSE 以 after=0 补齐——避免回放和实时流
  // 双通道向同一 run 交错推送（reducer 按 seq 判重时晚到的旧 seq 会被丢弃）。
  useEffect(() => {
    if (!taskId || !task.data) return;
    const pending = task.data.runs.filter(
      (run) => !loadedRuns.current.has(run.run_id) && run.run_id !== liveRunId,
    );
    if (pending.length === 0) {
      // 没有需要回放的 run：初始回放视为完成，放行 SSE
      setReplayDoneFor(taskId);
      return;
    }
    let cancelled = false;
    setReplaying(true);
    const replay = async () => {
      let sawArtifactEvent = false;
      for (const run of pending) {
        if (cancelled || loadedRuns.current.has(run.run_id)) continue;
        try {
          const snapshot = await api.getRunSnapshot(taskId, run.run_id);
          if (cancelled) return;
          const events = [...snapshot.events].sort((a, b) => a.seq - b.seq);
          const store = useTimelineStore.getState();
          if (store.taskId !== taskId) return;
          for (const event of events) {
            pushEvent(event);
            if (ARTIFACT_EVENT_TYPES.has(event.type)) sawArtifactEvent = true;
          }
          loadedRuns.current.add(run.run_id);
        } catch {
          // 旧版任务没有 stream.jsonl：标记跳过，避免每次轮询重试（详见 issues 文档）
          loadedRuns.current.add(run.run_id);
        }
      }
      if (cancelled) return;
      setReplaying(false);
      setReplayDoneFor(taskId);
      if (sawArtifactEvent) scheduleArtifactsRefetch();
    };
    void replay();
    return () => {
      cancelled = true;
      // 取消时复位骨架屏：若下一次任务详情里 pending 为空，replaying 不再被置回 true
      setReplaying(false);
    };
  }, [taskId, task.data, liveRunId, pushEvent, scheduleArtifactsRefetch]);

  // 实时流：初始回放完成后才订阅（after 取该 run 已处理的最大 seq，
  // 回放与订阅之间漏掉的事件由 reducer 按 seq 判重 + SSE 衔接兜底）；
  // 提交决策（sseEpoch 变化）强制重开同一 run 的订阅。
  useEffect(() => {
    if (!taskId || !liveRunId || replayDoneFor !== taskId) return;
    const after = useTimelineStore.getState().lastSeqByRun[liveRunId] ?? 0;
    let disposed = false;
    const close = subscribeRunEvents(taskId, liveRunId, after, {
      onEvent: (event) => {
        const store = useTimelineStore.getState();
        if (store.taskId !== taskId) return;
        store.pushEvent(event);
        // SSE 已推送的 run 不再走快照回放，避免重复拉取
        loadedRuns.current.add(event.run_id);
        if (ARTIFACT_EVENT_TYPES.has(event.type)) scheduleArtifactsRefetch();
      },
      onEnd: () => {
        if (disposed) return;
        setManual(null);
        void queryClient.invalidateQueries({ queryKey: ["task", taskId] });
        void queryClient.invalidateQueries({ queryKey: ["tasks"] });
      },
    });
    return () => {
      disposed = true;
      close();
    };
  }, [taskId, liveRunId, replayDoneFor, sseEpoch, queryClient, scheduleArtifactsRefetch]);

  const invalidateTask = useCallback(() => {
    if (!taskId) return;
    void queryClient.invalidateQueries({ queryKey: ["task", taskId] });
    void queryClient.invalidateQueries({ queryKey: ["tasks"] });
  }, [queryClient, taskId]);

  const startRun = useCallback(
    async (message: string, targetType: TargetType) => {
      if (!taskId) throw new Error("未选择任务");
      const { run_id } = await api.startRun(taskId, message, targetType);
      loadedRuns.current.add(run_id);
      setManual({ taskId, runId: run_id });
      invalidateTask();
    },
    [taskId, invalidateTask],
  );

  const submitReview = useCallback(
    async (runId: string, action: ReviewAction, feedback?: string) => {
      if (!taskId) return;
      await api.submitReview(taskId, runId, action, feedback);
      // 决策后事件继续落在同一个 run 的事件流上：重开 SSE 继续接收
      setManual({ taskId, runId });
      setSseEpoch((epoch) => epoch + 1);
      invalidateTask();
    },
    [taskId, invalidateTask],
  );

  const cancelRun = useCallback(() => {
    if (!taskId || !liveRunId) return;
    void api.cancelRun(taskId, liveRunId).catch(() => {
      // 取消失败时保持 SSE：结束事件仍会到达
    });
  }, [taskId, liveRunId]);

  const addFiles = useCallback(
    (files: File[]) => {
      if (!taskId || files.length === 0) return;
      const names = files.map((file) => file.name);
      setUploading((current) => [...current, ...names]);
      setUploadError(null);
      api
        .addSamples(taskId, files)
        .then(invalidateTask)
        .catch((cause: unknown) => {
          setUploadError(cause instanceof Error ? cause.message : "样本图上传失败");
        })
        .finally(() => {
          setUploading((current) => current.filter((name) => !names.includes(name)));
        });
    },
    [taskId, invalidateTask],
  );

  const removeSample = useMutation({
    mutationFn: (name: string) => api.removeSample(taskId!, name),
    onSuccess: invalidateTask,
  });

  const entries = useMemo(() => groupTimeline(items), [items]);

  // 最近一个未处理的 review（所在 run 停在 awaiting_review）驱动输入区形态
  const pendingReview = useMemo(() => {
    for (let index = items.length - 1; index >= 0; index -= 1) {
      const item = items[index];
      if (item.kind !== "review") continue;
      if (!item.resolved && runStatus[item.runId] === "awaiting_review") return item;
      return null;
    }
    return null;
  }, [items, runStatus]);

  // 每个 run 的原始任务描述（错误卡片“重试”用）
  const messageByRun = useMemo(() => {
    const map: Record<string, string> = {};
    for (const item of items) {
      if (item.kind === "user" && !(item.runId in map)) map[item.runId] = item.text;
    }
    return map;
  }, [items]);

  if (task.isPending) {
    return <div className="p-8 text-sm text-muted-foreground">加载任务…</div>;
  }
  if (task.isError || !task.data) {
    return (
      <div className="p-8 text-sm text-red-500">
        任务加载失败：{task.error instanceof Error ? task.error.message : "未知错误"}
      </div>
    );
  }

  const detail = task.data;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <TopBar task={detail} />
      <main
        className="min-h-0 flex-1"
        onDragOver={(event) => {
          if (event.dataTransfer.types.includes("Files")) event.preventDefault();
        }}
        onDrop={(event) => {
          if (!event.dataTransfer.types.includes("Files")) return;
          event.preventDefault();
          addFiles(Array.from(event.dataTransfer.files));
        }}
      >
        {replaying && entries.length === 0 ? (
          <div className="mx-auto max-w-3xl space-y-3 px-6 py-8">
            <Skeleton className="ml-auto h-10 w-64 rounded-2xl" />
            <Skeleton className="h-14 w-full rounded-xl" />
            <Skeleton className="h-14 w-5/6 rounded-xl" />
            <Skeleton className="h-40 w-full rounded-xl" />
          </div>
        ) : (
          <Timeline
            entries={entries}
            messageByRun={messageByRun}
            bestPipeline={detail.best?.pipeline ?? null}
            onOpenArtifacts={() => setRightPanel(true)}
            onReviewSubmit={submitReview}
            onRetry={(message) => void startRun(message, "defect")}
          />
        )}
      </main>
      {uploadError && (
        <p className="px-6 pb-1 text-xs text-red-600" role="alert">
          {uploadError}
        </p>
      )}
      <Composer
        samples={detail.samples}
        uploading={uploading}
        hasRuns={detail.runs.length > 0}
        running={running}
        pendingReview={pendingReview}
        onStart={startRun}
        onReview={(feedback) => {
          if (!pendingReview) throw new Error("没有待处理的确认");
          return submitReview(pendingReview.runId, "continue", feedback);
        }}
        onCancel={cancelRun}
        onAddFiles={addFiles}
        onRemoveSample={(name) => removeSample.mutate(name)}
      />
    </div>
  );
}
