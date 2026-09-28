// 事件 → 时间线条目的 reducer（计划 §9.2）。
// 必须是纯函数：同一事件序列重放结果完全一致（历史回放与实时流共用），
// 不读时钟、不读随机源；耗时一律来自事件自身字段（duration / ts 差值）。
import { create } from "zustand";
import type { ImageScore, ReviewStage, RunFinishStatus, UiEvent } from "@/api/types";

export type StepStatus = "running" | "done" | "failed";

export interface ModelCallChild {
  kind: "model_call";
  id: string;
  model: string;
  status: "running" | "done";
  output: string;
  usage?: Record<string, number> | null;
  contextWindow?: number | null;
  duration?: number | null;
}

export interface ToolChild {
  kind: "tool";
  id: string;
  tool: string;
  args: unknown;
  result?: unknown;
  success?: boolean;
  duration?: number | null;
}

export type StepChild = ModelCallChild | ToolChild;

export interface UserItem {
  kind: "user";
  id: string;
  runId: string;
  text: string;
}

export interface AssistantItem {
  kind: "assistant";
  id: string;
  runId: string;
  markdown: string;
}

export interface StepItem {
  kind: "step";
  id: string;
  runId: string;
  node: string;
  label: string;
  status: StepStatus;
  startedAt: number;
  duration?: number | null;
  thinking: string;
  children: StepChild[];
  metadata?: Record<string, unknown>;
}

export interface ReferenceItem {
  kind: "reference";
  id: string;
  runId: string;
  imageId: string;
  overlayUrl: string | null;
  samScore: number;
  lowQuality: boolean;
}

export interface IterationItem {
  kind: "iteration";
  id: string;
  runId: string;
  iteration: number;
  score: number;
  improved: boolean;
  pipeline: Record<string, unknown>[];
  notes: string;
  imageScores: ImageScore[];
}

export interface ReviewResolved {
  action: string;
  feedback?: string;
}

export interface ReviewItem {
  kind: "review";
  id: string;
  runId: string;
  stage: ReviewStage;
  message: string;
  overlayUrl: string | null;
  bestScore: number;
  stopReason?: string | null;
  resolved?: ReviewResolved;
}

export interface ErrorItem {
  kind: "error";
  id: string;
  runId: string;
  message: string;
  node?: string | null;
}

export interface RunEndItem {
  kind: "run_end";
  id: string;
  runId: string;
  status: RunFinishStatus;
  duration: number;
  bestScore?: number;
}

export type TimelineItem =
  | UserItem
  | AssistantItem
  | StepItem
  | ReferenceItem
  | IterationItem
  | ReviewItem
  | ErrorItem
  | RunEndItem;

export interface TimelineState {
  items: TimelineItem[];
  /** 每个 run 已处理的最大 seq；重连补发 / 快照重放时按 run 判重（计划 §9.2）。 */
  lastSeqByRun: Record<string, number>;
  /** 每个 run 最近一次 run_finished 的状态。 */
  runStatus: Record<string, RunFinishStatus>;
}

export function createTimelineState(): TimelineState {
  return { items: [], lastSeqByRun: {}, runStatus: {} };
}

// --- 查找辅助：全部基于 items 扫描，不维护额外配对状态，保证重放一致 ---

function implicitStepId(runId: string): string {
  return `implicit:${runId}`;
}

function findItemIndex(items: TimelineItem[], id: string): number {
  return items.findIndex((item) => item.id === id);
}

function lastRunningStepIndex(items: TimelineItem[], runId: string): number {
  // 返回同 run 中最靠后的运行中步骤（嵌套时即最内层）
  for (let index = items.length - 1; index >= 0; index -= 1) {
    const item = items[index];
    if (item.kind === "step" && item.runId === runId && item.status === "running") {
      return index;
    }
  }
  return -1;
}

interface ChildLocation {
  itemIndex: number;
  childIndex: number;
}

function findChildLocation(
  items: TimelineItem[],
  childId: string,
): ChildLocation | null {
  for (let itemIndex = items.length - 1; itemIndex >= 0; itemIndex -= 1) {
    const item = items[itemIndex];
    if (item.kind !== "step") continue;
    for (let childIndex = item.children.length - 1; childIndex >= 0; childIndex -= 1) {
      if (item.children[childIndex].id === childId) return { itemIndex, childIndex };
    }
  }
  return null;
}

function withItems(state: TimelineState, items: TimelineItem[]): TimelineState {
  return { ...state, items };
}

function cloneStep(items: TimelineItem[], index: number): StepItem {
  const step = { ...(items[index] as StepItem) };
  step.children = [...step.children];
  items[index] = step;
  return step;
}

/** 没有运行中 step 时，thinking / 模型 / 工具挂到隐式的“准备”步骤（计划 §9.2）。 */
function ensureImplicitStep(state: TimelineState, runId: string, ts: number): TimelineState {
  if (lastRunningStepIndex(state.items, runId) >= 0) return state;
  const items = [...state.items];
  // 同一 run 只保留一个隐式步骤：出现在任何真实步骤之前
  for (let index = items.length - 1; index >= 0; index -= 1) {
    const item = items[index];
    if (item.kind === "step" && item.id === implicitStepId(runId)) return state;
  }
  items.push({
    kind: "step",
    id: implicitStepId(runId),
    runId,
    node: "",
    label: "准备",
    status: "running",
    startedAt: ts,
    thinking: "",
    children: [],
  });
  return withItems(state, items);
}

function closeImplicitStep(state: TimelineState, runId: string, ts: number): TimelineState {
  const index = findItemIndex(state.items, implicitStepId(runId));
  if (index < 0) return state;
  const items = [...state.items];
  const step = cloneStep(items, index);
  if (step.status === "running") {
    step.status = "done";
    step.duration = Math.max(0, ts - step.startedAt);
    items[index] = step;
  }
  return withItems(state, items);
}

/** run 结束时收敛所有仍打开的 step：失败/取消标 failed，其余视为已停（计划 §9.2 只有三种状态）。 */
function closeOpenSteps(
  state: TimelineState,
  runId: string,
  ts: number,
  failed: boolean,
): TimelineState {
  let items: TimelineItem[] | null = null;
  for (let index = 0; index < state.items.length; index += 1) {
    const item = state.items[index];
    if (item.kind !== "step" || item.runId !== runId || item.status !== "running") continue;
    if (items === null) items = [...state.items];
    const step = { ...item };
    step.status = failed ? "failed" : "done";
    step.duration = Math.max(0, ts - item.startedAt);
    items[index] = step;
  }
  return items === null ? state : withItems(state, items);
}

function resolveThinkingTarget(
  state: TimelineState,
  runId: string,
  stepId: string | null,
  ts: number,
): { state: TimelineState; index: number } {
  if (stepId) {
    const index = findItemIndex(state.items, stepId);
    if (index >= 0 && state.items[index].kind === "step") return { state, index };
  }
  const running = lastRunningStepIndex(state.items, runId);
  if (running >= 0) return { state, index: running };
  const ensured = ensureImplicitStep(state, runId, ts);
  return { state: ensured, index: findItemIndex(ensured.items, implicitStepId(runId)) };
}

/** §9.5：resumed 运行的 user 条目文案。 */
function resumedUserText(
  stage: ReviewStage | undefined,
  action: string | undefined,
  feedback: string | undefined,
): string {
  const note = feedback ? `：${feedback}` : "";
  if (action === "exit") return "结束任务";
  if (action === "accept") return "接受当前最优结果";
  if (action === "continue") {
    if (stage === "reference") {
      return feedback ? `修正参考掩膜：${feedback}` : "确认参考掩膜";
    }
    return `继续迭代${note}`;
  }
  return "继续执行";
}

// --- reducer 本体 ---

export function applyEvent(state: TimelineState, event: UiEvent): TimelineState {
  const lastSeq = state.lastSeqByRun[event.run_id];
  if (lastSeq !== undefined && event.seq <= lastSeq) return state;
  let next: TimelineState = {
    ...state,
    lastSeqByRun: { ...state.lastSeqByRun, [event.run_id]: event.seq },
  };

  switch (event.type) {
    case "run_started": {
      const items = [...next.items];
      if (event.resumed) {
        // 把同一 run 里最近一个未处理的 review 标记为已响应（计划 §9.2）
        for (let index = items.length - 1; index >= 0; index -= 1) {
          const item = items[index];
          if (item.kind === "review" && item.runId === event.run_id && !item.resolved) {
            const resolved: ReviewResolved = { action: event.action ?? "" };
            if (event.feedback) resolved.feedback = event.feedback;
            items[index] = { ...item, resolved };
            next = withItems(next, items);
            next = pushItem(next, {
              kind: "user",
              id: `user:${event.run_id}:${event.seq}`,
              runId: event.run_id,
              text: resumedUserText(item.stage, event.action, event.feedback),
            });
            return next;
          }
        }
      }
      next = withItems(next, items);
      return pushItem(next, {
        kind: "user",
        id: `user:${event.run_id}:${event.seq}`,
        runId: event.run_id,
        text: event.message ?? "",
      });
    }

    case "step_started": {
      next = closeImplicitStep(next, event.run_id, event.ts);
      return pushItem(next, {
        kind: "step",
        id: event.step_id,
        runId: event.run_id,
        node: event.node,
        label: event.label,
        status: "running",
        startedAt: event.ts,
        thinking: "",
        children: [],
      });
    }

    case "step_finished": {
      const index = findItemIndex(next.items, event.step_id);
      if (index < 0) return next;
      const items = [...next.items];
      const step = { ...(items[index] as StepItem) };
      step.status = "done";
      step.duration = event.duration;
      step.metadata = event.metadata;
      items[index] = step;
      return withItems(next, items);
    }

    case "thinking": {
      const target = resolveThinkingTarget(next, event.run_id, event.step_id, event.ts);
      if (target.index < 0) return target.state;
      const items = [...target.state.items];
      const step = cloneStep(items, target.index);
      const glue = !event.delta && step.thinking ? "\n" : "";
      step.thinking = step.thinking + glue + event.text;
      return withItems(target.state, items);
    }

    case "model_call_started": {
      next = ensureImplicitStep(next, event.run_id, event.ts);
      const index = lastRunningStepIndex(next.items, event.run_id);
      if (index < 0) return next;
      const items = [...next.items];
      const step = cloneStep(items, index);
      step.children.push({
        kind: "model_call",
        id: event.call_id,
        model: event.model,
        status: "running",
        output: "",
      });
      return withItems(next, items);
    }

    case "model_call_finished": {
      const location = findChildLocation(next.items, event.call_id);
      if (!location) return next;
      const items = [...next.items];
      const step = cloneStep(items, location.itemIndex);
      const child = { ...(step.children[location.childIndex] as ModelCallChild) };
      child.status = "done";
      child.usage = event.usage;
      child.contextWindow = event.context_window;
      child.duration = event.duration;
      step.children[location.childIndex] = child;
      return withItems(next, items);
    }

    case "model_output": {
      let location = findChildLocation(next.items, event.call_id);
      if (!location) {
        // 翻译器把输出记在最近一个未完成调用上；找不到指定 id 时同样兜底
        const running = lastRunningStepIndex(next.items, event.run_id);
        if (running < 0) return next;
        const step = next.items[running] as StepItem;
        const childIndex = step.children.length - 1;
        if (childIndex < 0 || step.children[childIndex].kind !== "model_call") return next;
        location = { itemIndex: running, childIndex };
      }
      const items = [...next.items];
      const step = cloneStep(items, location.itemIndex);
      const child = { ...(step.children[location.childIndex] as ModelCallChild) };
      child.output += event.text;
      step.children[location.childIndex] = child;
      return withItems(next, items);
    }

    case "tool_started": {
      next = ensureImplicitStep(next, event.run_id, event.ts);
      const index = lastRunningStepIndex(next.items, event.run_id);
      if (index < 0) return next;
      const items = [...next.items];
      const step = cloneStep(items, index);
      step.children.push({
        kind: "tool",
        id: event.tool_id,
        tool: event.tool,
        args: event.args,
      });
      return withItems(next, items);
    }

    case "tool_finished": {
      const location = findChildLocation(next.items, event.tool_id);
      if (!location) return next;
      const items = [...next.items];
      const step = cloneStep(items, location.itemIndex);
      const child = { ...(step.children[location.childIndex] as ToolChild) };
      child.result = event.result;
      child.success = event.success;
      child.duration = event.duration;
      step.children[location.childIndex] = child;
      return withItems(next, items);
    }

    case "reference_candidate": {
      return pushItem(next, {
        kind: "reference",
        id: `reference:${event.run_id}:${event.seq}`,
        runId: event.run_id,
        imageId: event.image_id,
        overlayUrl: event.overlay_url,
        samScore: event.sam_score,
        lowQuality: event.low_quality,
      });
    }

    case "iteration_scored": {
      return pushItem(next, {
        kind: "iteration",
        id: `iteration:${event.run_id}:${event.seq}`,
        runId: event.run_id,
        iteration: event.iteration,
        score: event.composite_mean,
        improved: event.improved,
        pipeline: event.pipeline,
        notes: event.notes,
        imageScores: event.image_scores,
      });
    }

    case "review_requested": {
      return pushItem(next, {
        kind: "review",
        id: `review:${event.run_id}:${event.seq}`,
        runId: event.run_id,
        stage: event.stage,
        message: event.message,
        overlayUrl: event.overlay_url,
        bestScore: event.best_score,
        stopReason: event.stop_reason ?? null,
      });
    }

    case "error": {
      // error 带节点时，把该 run 里运行中的同名节点标为失败
      if (event.node) {
        next = closeOpenStepsForNode(next, event.run_id, event.node);
      }
      return pushItem(next, {
        kind: "error",
        id: `error:${event.run_id}:${event.seq}`,
        runId: event.run_id,
        message: event.message,
        node: event.node ?? null,
      });
    }

    case "run_finished": {
      next = closeOpenSteps(next, event.run_id, event.ts, event.status === "failed" || event.status === "cancelled");
      const runEnd: RunEndItem = {
        kind: "run_end",
        id: `run_end:${event.run_id}:${event.seq}`,
        runId: event.run_id,
        status: event.status,
        duration: event.duration,
      };
      if (event.best_score !== undefined) runEnd.bestScore = event.best_score;
      next = pushItem(next, runEnd);
      return { ...next, runStatus: { ...next.runStatus, [event.run_id]: event.status } };
    }

    default:
      return next;
  }
}

function closeOpenStepsForNode(
  state: TimelineState,
  runId: string,
  node: string,
): TimelineState {
  let items: TimelineItem[] | null = null;
  for (let index = 0; index < state.items.length; index += 1) {
    const item = state.items[index];
    if (item.kind !== "step" || item.runId !== runId || item.status !== "running") continue;
    if (item.node !== node) continue;
    if (items === null) items = [...state.items];
    const step = { ...item };
    step.status = "failed";
    items[index] = step;
  }
  return items === null ? state : withItems(state, items);
}

function pushItem(state: TimelineState, item: TimelineItem): TimelineState {
  return withItems(state, [...state.items, item]);
}

export function applyEvents(state: TimelineState, events: UiEvent[]): TimelineState {
  return events.reduce((current, event) => applyEvent(current, event), state);
}

// --- 迭代折叠分组（计划 §9.2：连续的迭代条目默认折叠成一组）---
// 迭代轮之间夹着 iterate/score/promote 步骤卡片，因此把「连续的 step+iteration
// 片段且含至少一个 iteration」折叠为一组；标题显示 轮次区间和最优分数。

export interface IterationGroupEntry {
  kind: "iteration_group";
  id: string;
  items: TimelineItem[];
  from: number;
  to: number;
  best: number;
}

export type TimelineEntry =
  | { kind: "item"; item: TimelineItem }
  | IterationGroupEntry;

export function groupTimeline(items: TimelineItem[]): TimelineEntry[] {
  const entries: TimelineEntry[] = [];
  let buffer: TimelineItem[] = [];

  const flush = () => {
    if (buffer.length === 0) return;
    const iterations = buffer.filter(
      (item): item is IterationItem => item.kind === "iteration",
    );
    if (iterations.length > 0) {
      entries.push({
        kind: "iteration_group",
        id: `iteration_group:${iterations[0].id}`,
        items: buffer,
        from: iterations[0].iteration,
        to: iterations[iterations.length - 1].iteration,
        best: Math.max(...iterations.map((item) => item.score)),
      });
    } else {
      for (const item of buffer) entries.push({ kind: "item", item });
    }
    buffer = [];
  };

  for (const item of items) {
    if (item.kind === "step" || item.kind === "iteration") {
      buffer.push(item);
    } else {
      flush();
      entries.push({ kind: "item", item });
    }
  }
  flush();
  return entries;
}

// --- zustand 封装：当前任务的时间线状态（事件驱动的部分全在纯函数里）---

interface TimelineStore extends TimelineState {
  taskId: string | null;
  beginTask: (taskId: string) => void;
  pushEvent: (event: UiEvent) => void;
}

export const useTimelineStore = create<TimelineStore>((set) => ({
  ...createTimelineState(),
  taskId: null,
  beginTask: (taskId) => set({ ...createTimelineState(), taskId }),
  pushEvent: (event) => set((state) => applyEvent(state, event)),
}));
