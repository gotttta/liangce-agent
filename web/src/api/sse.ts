import type { UiEvent } from "./types";

// SSE 订阅封装（计划 §9.8）：原生 EventSource，断线由浏览器自动重连并携带
// Last-Event-ID；服务端发 event: end 表示该运行（或该段执行）结束，主动关闭。

export interface RunEventHandlers {
  onEvent: (event: UiEvent) => void;
  /** 收到 event: end：本段执行结束（awaiting_review 时等待用户决策）。 */
  onEnd: () => void;
}

export function subscribeRunEvents(
  taskId: string,
  runId: string,
  afterSeq: number,
  handlers: RunEventHandlers,
): () => void {
  const url = `/api/tasks/${encodeURIComponent(taskId)}/runs/${encodeURIComponent(runId)}/events?after=${afterSeq}`;
  const source = new EventSource(url);
  let ended = false;

  source.addEventListener("ui", (raw) => {
    const data = (raw as MessageEvent<string>).data;
    try {
      handlers.onEvent(JSON.parse(data) as UiEvent);
    } catch {
      // 单条坏帧不应断开整条流：交给 reducer 按 seq 判重兜底
    }
  });
  source.addEventListener("end", () => {
    ended = true;
    source.close();
    handlers.onEnd();
  });
  // onerror 时浏览器会自动重连（携带 Last-Event-ID 头，服务端优先采用），
  // 这里不主动 close；只有任务切换 / 组件卸载会调用返回的清理函数。
  return () => {
    if (!ended) source.close();
  };
}
