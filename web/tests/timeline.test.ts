// 时间线 reducer 测试（计划 §阶段5）：fixture 由阶段 3 的 fake 工作流实际
// 产出（tests/fixtures/export_run_events.py），保证前后端协议一致。
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import type { UiEvent } from "@/api/types";
import {
  applyEvent,
  applyEvents,
  createTimelineState,
  groupTimeline,
  type IterationItem,
  type ReviewItem,
  type StepItem,
  type UserItem,
} from "@/store/timeline";

const fixtureDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures");
const fixture = JSON.parse(
  readFileSync(path.join(fixtureDir, "run_events.json"), "utf-8"),
) as UiEvent[];

function replay(events: UiEvent[] = fixture) {
  return applyEvents(createTimelineState(), events);
}

describe("fixture 回放", () => {
  it("首条是用户消息，事件全部按序处理", () => {
    const state = replay();
    expect(state.items[0]).toMatchObject({
      kind: "user",
      text: "找出图中的亮块",
    });
    expect(state.lastSeqByRun[fixture[0].run_id]).toBe(fixture[fixture.length - 1].seq);
    expect(state.runStatus[fixture[0].run_id]).toBe("completed");
  });

  it("step 全部关闭：重放后没有 running 状态的步骤", () => {
    const state = replay();
    const steps = state.items.filter((item): item is StepItem => item.kind === "step");
    expect(steps.length).toBeGreaterThan(0);
    for (const step of steps) expect(step.status).not.toBe("running");
  });

  it("thinking 与模型调用挂到当前运行的 step 下", () => {
    const state = replay();
    const prepare = state.items.find(
      (item): item is StepItem => item.kind === "step" && item.id === "node:prepare:1",
    );
    expect(prepare?.thinking).toContain("样本共 2 张");
    const gen = state.items.find(
      (item): item is StepItem => item.kind === "step" && item.id === "node:gen_reference:1",
    );
    const modelCall = gen?.children.find((child) => child.kind === "model_call");
    expect(modelCall).toMatchObject({ model: "qwen-vl-max", status: "done" });
    expect(modelCall).toHaveProperty("usage");
  });

  it("参考确认卡片带叠加图 URL 与低质量标记", () => {
    const state = replay();
    const references = state.items.filter((item) => item.kind === "reference");
    expect(references).toHaveLength(2);
    expect(references[0]).toMatchObject({
      imageId: "img_a",
      overlayUrl: "/api/files?path=outputs/overlay_img_a.png",
      lowQuality: false,
    });
    expect(references[1]).toMatchObject({ imageId: "img_b", lowQuality: true });
  });

  it("三次人工确认里前两次被后续 resumed 运行标记为已响应", () => {
    const state = replay();
    const reviews = state.items.filter((item): item is ReviewItem => item.kind === "review");
    expect(reviews).toHaveLength(3);
    expect(reviews.map((review) => review.stage)).toEqual(["reference", "reference", "final"]);
    expect(reviews[0].resolved).toEqual({
      action: "continue",
      feedback: "边缘偏小，请包含完整的亮块",
    });
    expect(reviews[1].resolved).toEqual({ action: "continue" });
    expect(reviews[2].resolved).toEqual({ action: "accept" });
    expect(reviews[2].stopReason).toBe("target_reached");
  });

  it("resumed 运动生成 §9.5 文案的用户消息", () => {
    const state = replay();
    const users = state.items.filter((item): item is UserItem => item.kind === "user");
    expect(users.map((user) => user.text)).toEqual([
      "找出图中的亮块",
      "修正参考掩膜：边缘偏小，请包含完整的亮块",
      "确认参考掩膜",
      "接受当前最优结果",
    ]);
  });
});

describe("重复 seq 忽略与重放一致", () => {
  it("同一序列重放两遍，第二遍全部被忽略", () => {
    const once = replay();
    const twice = applyEvents(once, fixture);
    expect(twice.items).toHaveLength(once.items.length);
    expect(twice.lastSeqByRun).toEqual(once.lastSeqByRun);
  });

  it("分批重放与一次重放结果完全一致", () => {
    const whole = replay();
    const split = applyEvents(applyEvents(createTimelineState(), fixture.slice(0, 30)), fixture.slice(30));
    expect(split).toEqual(whole);
  });

  it("乱序重放（先旧后新）不产生重复条目", () => {
    const state = replay();
    const reordered = applyEvents(createTimelineState(), [...fixture].reverse());
    // seq 乱序时按 run 判重只保留更新的事件，条目数不膨胀
    expect(reordered.items.length).toBeLessThanOrEqual(state.items.length);
  });
});

describe("迭代折叠分组", () => {
  it("连续的迭代（连同中间的步骤卡片）折叠成一组", () => {
    const state = replay();
    const entries = groupTimeline(state.items);
    const groups = entries.filter((entry) => entry.kind === "iteration_group");
    expect(groups).toHaveLength(1);
    const group = groups[0];
    if (group.kind !== "iteration_group") throw new Error("unreachable");
    expect(group.from).toBe(1);
    expect(group.to).toBe(3);
    expect(group.best).toBeCloseTo(0.71, 5);
    const iterations = group.items.filter(
      (item): item is IterationItem => item.kind === "iteration",
    );
    expect(iterations).toHaveLength(3);
    // 迭代条目全部进组，不会以单条形式散落在时间线上
    for (const entry of entries) {
      if (entry.kind === "item") {
        expect(entry.item.kind).not.toBe("iteration");
      }
    }
    // 用户消息、参考卡片、确认卡片不被折叠
    const individualKinds = entries
      .filter((entry) => entry.kind === "item")
      .map((entry) => (entry.kind === "item" ? entry.item.kind : ""));
    expect(individualKinds).toContain("user");
    expect(individualKinds).toContain("reference");
    expect(individualKinds).toContain("review");
  });

  it("不含迭代的步骤序列不折叠", () => {
    const state = replay();
    const entries = groupTimeline(
      state.items.filter((item) => item.kind !== "iteration"),
    );
    expect(entries.every((entry) => entry.kind === "item")).toBe(true);
  });
});

describe("手工事件：嵌套步骤与隐式准备步骤", () => {
  const run = "agent_test";

  function base(seq: number): { seq: number; run_id: string; ts: number } {
    return { seq, run_id: run, ts: 1_000 + seq };
  }

  it("嵌套步骤中的 thinking / 工具挂到最内层运行步骤", () => {
    let state = createTimelineState();
    state = applyEvent(state, { ...base(1), type: "step_started", step_id: "node:a:1", node: "a", label: "A" });
    state = applyEvent(state, { ...base(2), type: "step_started", step_id: "node:b:1", node: "b", label: "B" });
    state = applyEvent(state, { ...base(3), type: "thinking", step_id: null, context: null, text: "内层思考", delta: true });
    state = applyEvent(state, { ...base(4), type: "tool_started", tool_id: "tool:1", tool: "t", args: {} });
    state = applyEvent(state, { ...base(5), type: "step_finished", step_id: "node:b:1", duration: 2, metadata: {} });
    state = applyEvent(state, { ...base(6), type: "thinking", step_id: null, context: null, text: "外层思考", delta: true });

    const outer = state.items.find((item): item is StepItem => item.kind === "step" && item.id === "node:a:1");
    const inner = state.items.find((item): item is StepItem => item.kind === "step" && item.id === "node:b:1");
    expect(inner?.status).toBe("done");
    expect(inner?.thinking).toBe("内层思考");
    expect(inner?.children).toHaveLength(1);
    expect(outer?.status).toBe("running");
    expect(outer?.thinking).toBe("外层思考");
    expect(outer?.children).toHaveLength(0);
  });

  it("没有运行中步骤时挂到隐式“准备”步骤，真实步骤开始后关闭", () => {
    let state = createTimelineState();
    state = applyEvent(state, { ...base(1), type: "thinking", step_id: null, context: null, text: "预热", delta: true });
    const implicit = state.items.find((item): item is StepItem => item.kind === "step");
    expect(implicit).toMatchObject({ label: "准备", status: "running", thinking: "预热" });
    state = applyEvent(state, { ...base(2), type: "step_started", step_id: "node:a:1", node: "a", label: "A" });
    const closed = state.items.find((item): item is StepItem => item.kind === "step" && item.id === implicit?.id);
    expect(closed?.status).toBe("done");
    expect(closed?.duration).toBe(1);
  });

  it("run_finished 收敛仍打开的步骤：取消视为失败，其余视为已完成", () => {
    let cancelled = createTimelineState();
    cancelled = applyEvent(cancelled, { ...base(1), type: "step_started", step_id: "node:a:1", node: "a", label: "A" });
    cancelled = applyEvent(cancelled, { ...base(2), type: "run_finished", status: "cancelled", duration: 3 });
    const step = cancelled.items.find((item): item is StepItem => item.kind === "step");
    expect(step?.status).toBe("failed");

    let completed = createTimelineState();
    completed = applyEvent(completed, { ...base(1), type: "step_started", step_id: "node:a:1", node: "a", label: "A" });
    completed = applyEvent(completed, { ...base(2), type: "run_finished", status: "awaiting_review", duration: 3 });
    const gate = completed.items.find((item): item is StepItem => item.kind === "step");
    expect(gate?.status).toBe("done");
    expect(completed.runStatus[run]).toBe("awaiting_review");
  });

  it("error 事件把同名运行中节点标记为失败并追加错误条目", () => {
    let state = createTimelineState();
    state = applyEvent(state, { ...base(1), type: "step_started", step_id: "node:score:1", node: "score", label: "打分" });
    state = applyEvent(state, { ...base(2), type: "error", message: "pipeline operator requires v3 named inputs", node: "score" });
    const step = state.items.find((item): item is StepItem => item.kind === "step");
    expect(step?.status).toBe("failed");
    const error = state.items.find((item) => item.kind === "error");
    expect(error).toMatchObject({ message: "pipeline operator requires v3 named inputs" });
  });
});
