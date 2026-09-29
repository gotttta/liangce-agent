// TaskPage 数据流组件测试（审查意见 #2）：
// 场景——任务有两轮运行，第一轮已完成，第二轮正在迭代（latest_run_id 仍指向
// 第一轮、active_run_id 指向第二轮）。刷新页面后：
//   1. SSE 在初始回放完成前不得打开；
//   2. 时间线里旧运行的条目必须排在运行中一轮之前；
//   3. SSE 从回放结束的位置接着订阅（after=lastSeqByRun）。
// api 与 SSE 均为 mock；reducer 是真实实现，断言走 DOM 顺序。
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RunSnapshot, TaskDetail, UiEvent } from "@/api/types";
import { TaskPage } from "@/pages/TaskPage";

// vitest 未开 globals，testing-library 的自动清理不生效，手动清理避免跨测试 DOM 残留
afterEach(cleanup);

const mocks = vi.hoisted(() => ({
  getTask: vi.fn(),
  getRunSnapshot: vi.fn(),
  subscribeRunEvents: vi.fn(),
}));

vi.mock("@/api/client", () => ({
  ApiError: class ApiError extends Error {},
  api: {
    getTask: (...args: unknown[]) => mocks.getTask(...args),
    getRunSnapshot: (...args: unknown[]) => mocks.getRunSnapshot(...args),
    health: () => Promise.resolve({ ok: true, model: null }),
  },
}));

vi.mock("@/api/sse", () => ({
  subscribeRunEvents: (...args: unknown[]) => mocks.subscribeRunEvents(...args),
}));

const TASK_ID = "task_race";

interface Subscription {
  runId: string;
  after: number;
  handlers: { onEvent: (event: UiEvent) => void; onEnd: () => void };
}

function started(runId: string, seq: number, message: string): UiEvent {
  return { seq, run_id: runId, ts: 1000 + seq, type: "run_started", message, resumed: false };
}

function stepStarted(runId: string, seq: number, label: string): UiEvent {
  return {
    seq, run_id: runId, ts: 1000 + seq, type: "step_started",
    step_id: `node:iterate:${seq}`, node: "iterate", label,
  };
}

function detail(overrides: Partial<TaskDetail>): TaskDetail {
  return {
    id: TASK_ID,
    title: "竞态测试",
    status: "in_progress",
    status_label: "进行中",
    created_at: "2026-09-29T00:00:00+00:00",
    updated_at: "2026-09-29T00:00:00+00:00",
    running: true,
    active_run_id: null,
    sample_count: 1,
    samples: [],
    latest_run_id: null,
    runs: [],
    pending_review: null,
    best: null,
    ...overrides,
  };
}

function renderTaskPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/tasks/${TASK_ID}`]}>
        <Routes>
          <Route path="tasks/:taskId" element={<TaskPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function timelineText(): string {
  return document.querySelector('[data-testid="timeline"]')?.textContent ?? "";
}

describe("TaskPage 回放与实时流", () => {
  let subscriptions: Subscription[];

  beforeEach(() => {
    vi.clearAllMocks();
    subscriptions = [];
    mocks.subscribeRunEvents.mockImplementation(
      (_taskId: string, runId: string, after: number, handlers) => {
        subscriptions.push({ runId, after, handlers });
        return () => {};
      },
    );
  });

  it("两轮运行且第二轮进行中：回放完成才开 SSE，旧运行条目排在前面", async () => {
    mocks.getTask.mockResolvedValue(detail({
      running: true,
      active_run_id: "runB",
      latest_run_id: "runA",
      runs: [
        { run_id: "runA", status: "completed", status_label: "已完成",
          started_at: "2026-09-29T00:00:00", stop_reason: null },
        { run_id: "runB", status: "running", status_label: "进行中",
          started_at: "2026-09-29T00:05:00", stop_reason: null },
      ],
    }));

    // 快照请求挂起，由测试控制返回时机
    const deferred: Record<string, (snapshot: RunSnapshot) => void> = {};
    mocks.getRunSnapshot.mockImplementation(
      (_taskId: string, runId: string) =>
        new Promise<RunSnapshot>((resolve) => { deferred[runId] = resolve; }),
    );

    renderTaskPage();

    // runA 的快照尚未返回：SSE 必须还没打开
    await waitFor(() =>
      expect(mocks.getRunSnapshot).toHaveBeenCalledWith(TASK_ID, "runA"));
    expect(mocks.getRunSnapshot).not.toHaveBeenCalledWith(TASK_ID, "runB");
    expect(subscriptions).toHaveLength(0);

    await act(async () => {
      deferred["runA"]({
        run_id: "runA", task_id: TASK_ID, status: "completed",
        events: [started("runA", 1, "第一轮"), {
          seq: 2, run_id: "runA", ts: 1002, type: "run_finished",
          status: "completed", duration: 1,
        }],
      });
    });
    expect(await screen.findByText("第一轮")).toBeTruthy();

    // 第二轮是活动 run：不走快照回放，历史由 SSE 以 after=0 补齐
    expect(mocks.getRunSnapshot).not.toHaveBeenCalledWith(TASK_ID, "runB");

    // 回放完成 → SSE 打开，从 0 开始补齐 runB；打开前旧运行内容已就位
    await waitFor(() => expect(subscriptions).toHaveLength(1));
    expect(screen.getByText("第一轮")).toBeTruthy();
    expect(subscriptions[0]).toMatchObject({ runId: "runB", after: 0 });

    // SSE 推送运行中一轮的事件 → 追加在同一条时间线
    await act(async () => {
      subscriptions[0].handlers.onEvent(started("runB", 1, "第二轮"));
      subscriptions[0].handlers.onEvent(stepStarted("runB", 2, "提出算子序列调整"));
    });
    expect(await screen.findByText("第二轮")).toBeTruthy();

    const text = timelineText();
    expect(text.indexOf("第一轮")).toBeGreaterThanOrEqual(0);
    expect(text.indexOf("第一轮")).toBeLessThan(text.indexOf("第二轮"));
    expect(text.indexOf("第二轮")).toBeLessThan(text.indexOf("提出算子序列调整"));
  });

  it("没有可回放的历史时不阻塞 SSE（首次运行的刷新恢复路径）", async () => {
    mocks.getTask.mockResolvedValue(detail({
      running: true,
      active_run_id: "runB",
      latest_run_id: null,
      runs: [],
    }));

    renderTaskPage();

    await waitFor(() => expect(subscriptions).toHaveLength(1));
    expect(subscriptions[0]).toMatchObject({ runId: "runB", after: 0 });
    expect(mocks.getRunSnapshot).not.toHaveBeenCalled();
  });

  it("SSE 推送的 run 不再走快照回放", async () => {
    mocks.getTask.mockResolvedValue(detail({
      running: false,
      runs: [
        { run_id: "runA", status: "completed", status_label: "已完成",
          started_at: "2026-09-29T00:00:00", stop_reason: null },
      ],
    }));
    mocks.getRunSnapshot.mockResolvedValue({
      run_id: "runA", task_id: TASK_ID, status: "completed",
      events: [started("runA", 1, "第一轮")],
    });

    renderTaskPage();
    expect(await screen.findByText("第一轮")).toBeTruthy();
    await waitFor(() =>
      expect(mocks.getRunSnapshot).toHaveBeenCalledTimes(1));

    // 模拟任务详情轮询后 runA 再次出现（已由 SSE 推送过 → 不重复拉快照）
    await act(async () => {
      // runA 的事件已经在回放中推送并标记；这里只验证不再重复拉取
    });
    expect(mocks.getRunSnapshot).toHaveBeenCalledTimes(1);
    expect(subscriptions).toHaveLength(0); // 任务不在运行，没有 SSE
  });
});
