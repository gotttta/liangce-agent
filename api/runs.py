"""后台运行管理（计划 §7）：运行生命周期与 HTTP 连接解耦，SSE 只是订阅。

- 每次 start/resume 开一个 daemon 线程；断开 SSE、断开页面都不影响运行，
  只有 cancel 会设置 request_control.cancelled。
- listener 必须用 run_id 登记（子线程模型调用可能不在同一线程）；
  RunManager 预生成 run_id 并通过 thread_id= 传进 run_agent_graph，
  worker 里先 bind_context(run_id=...)，否则 logged_operation 会随机生成 run_id。
- 事件：单调递增 seq（每 run 独立，resume 接着计）、150ms 增量合并、
  stream.jsonl 持久化（写盘前 redact）。
"""
import contextvars
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from core.agent_events import register_event_listener, unregister_event_listener
from core.orchestration_runtime import TaskBusyError
from core.request_control import RequestCancelled, RequestControl, control
from core.runtime_logging import bind_context, logger, redact

from api.events import EventTranslator
from api.files import FileRoots, file_url

RETENTION_SECONDS = 600          # 运行结束后内存缓冲保留时长
MERGE_WINDOW = 0.15              # 流式增量合并窗口（秒）
SUBSCRIBE_WAIT = 15.0            # 订阅等待超时，超时发 SSE 注释行保活
MERGEABLE_TYPES = {"thinking", "model_output"}


class RunHandle:
    def __init__(self, task_id: str, run_id: str, stream_path: Path):
        self.task_id = task_id
        self.run_id = run_id
        self.stream_path = stream_path
        self.translator: EventTranslator | None = None  # 由 RunManager 注入（带 roots）
        self.events: list[dict] = []
        self.seq = 0
        self.pending: dict | None = None      # 合并窗口内的流式增量
        self.pending_since = 0.0
        self.condition = threading.Condition()
        self.finished = False
        self.finished_at: float | None = None
        self.status = "unknown"               # running/终态；unknown=从文件恢复
        self.request_control: RequestControl | None = None


def _redact_value(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        # dict 走 JSON 往返，让 redact 的 "api_key=" 类模式能命中键值对
        try:
            return json.loads(redact(json.dumps(value, ensure_ascii=False, default=str)))
        except ValueError:
            return redact(str(value))
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _redact_event(event: dict) -> dict:
    return {key: _redact_value(value) for key, value in event.items()}


def sse_frame(event: dict) -> str:
    return (f"id: {event['seq']}\nevent: ui\n"
            f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n")


class RunManager:
    def __init__(self, task_store, output_root: Path, roots: FileRoots,
                 run_fn=None, resume_fn=None, provider_factory=None):
        self.task_store = task_store
        self.output_root = Path(output_root)
        self.roots = roots
        self._run_fn = run_fn
        self._resume_fn = resume_fn
        self._provider_factory = provider_factory
        self._handles: dict[tuple[str, str], RunHandle] = {}
        self._active: dict[str, str] = {}     # task_id -> run_id
        self._lock = threading.Lock()

    # --- 对外接口 ---

    def start(self, task_id: str, message: str, target_type: str = "defect") -> str:
        task = self.task_store.load_task(task_id)
        samples = [str(sample["path"]) for sample in task.get("samples") or []]
        if not samples:
            raise ValueError("请先上传样本图")
        with self._lock:
            self._sweep_locked()
            if task_id in self._active:
                raise TaskBusyError(f"task {task_id} already has an active runner")
            run_id = f"agent_{uuid4().hex}"
            handle = self._new_handle_locked(task_id, run_id)
            handle.status = "running"
            self._active[task_id] = run_id
        self._spawn(task_id, run_id, SimpleNamespace(
            kind="start", message=message, target_type=target_type, samples=samples))
        return run_id

    def resume(self, task_id: str, run_id: str, action: str, feedback: str = "") -> None:
        with self._lock:
            self._sweep_locked()
            active = self._active.get(task_id)
            if active is not None and active != run_id:
                raise TaskBusyError(f"task {task_id} already has an active runner")
            handle = self._get_or_restore_locked(task_id, run_id)
            handle.status = "running"
            self._active[task_id] = run_id
        self._spawn(task_id, run_id, SimpleNamespace(
            kind="resume", action=action, feedback=feedback or ""))

    def cancel(self, task_id: str, run_id: str) -> bool:
        with self._lock:
            handle = self._handles.get((task_id, run_id))
        if handle is None:
            self._raise_or_missing(run_id)
        if handle.request_control is not None:
            handle.request_control.cancelled.set()
        return True

    def is_running(self, task_id: str) -> bool:
        with self._lock:
            self._sweep_locked()
            return task_id in self._active

    def subscribe(self, task_id: str, run_id: str, after_seq: int = 0):
        """返回 SSE 生成器；先补发 seq > after_seq 的历史，再跟随实时，结束发 end。"""
        with self._lock:
            handle = self._get_or_restore_locked(task_id, run_id)
        return self._stream(handle, int(after_seq))

    def snapshot(self, task_id: str, run_id: str) -> dict:
        with self._lock:
            handle = self._get_or_restore_locked(task_id, run_id)
        with handle.condition:
            events = list(handle.events)
            status = handle.status
        if status == "unknown":
            status = "interrupted"   # 无活动 worker 又没有终态：进程重启过的运行
            for event in reversed(events):
                if event.get("type") == "run_finished":
                    status = str(event.get("status") or "interrupted")
                    break
        return {"run_id": run_id, "task_id": task_id, "status": status, "events": events}

    # --- 后台执行 ---

    def _spawn(self, task_id: str, run_id: str, payload) -> None:
        worker_context = contextvars.copy_context()
        thread = threading.Thread(
            target=worker_context.run,
            args=(lambda: self._worker(task_id, run_id, payload),),
            name=f"run-{run_id}", daemon=True)
        thread.start()

    def _worker(self, task_id: str, run_id: str, payload) -> None:
        with self._lock:
            handle = self._handles[(task_id, run_id)]
        started = time.monotonic()
        request_control = RequestControl.from_env()
        token = control.set(request_control)
        handle.request_control = request_control
        bind_context(task_id=task_id, run_id=run_id)  # 必须先于 run_agent_graph（setdefault）
        listener = self._make_listener(handle)
        register_event_listener(listener, run_id=run_id)
        try:
            if payload.kind == "start":
                self._append(handle, {
                    "type": "run_started", "message": payload.message,
                    "target_type": payload.target_type,
                    "image_count": len(payload.samples), "resumed": False})
                result = self._call_start(task_id, run_id, payload)
            else:
                started_event = {"type": "run_started", "resumed": True, "action": payload.action}
                if payload.feedback:
                    started_event["feedback"] = payload.feedback
                self._append(handle, started_event)
                result = self._call_resume(run_id, payload)
            interrupts = (result or {}).get("interrupt") or []
            status = "completed"
            if interrupts:
                self._append(handle, self._review_event(interrupts[0]))
                status = "awaiting_review"
            finished = {"type": "run_finished", "status": status,
                        "duration": round(time.monotonic() - started, 3)}
            best_score = float((result or {}).get("best_score") or 0.0)
            if best_score > 0:
                finished["best_score"] = best_score
            handle.status = status
            self._append(handle, finished)
        except RequestCancelled:
            handle.status = "cancelled"
            self._append(handle, {"type": "run_finished", "status": "cancelled",
                                  "duration": round(time.monotonic() - started, 3)})
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            handle.status = "failed"
            # §8：error 事件的 message 必须已过滤（SSE 与落盘同源）
            self._append(handle, {"type": "error",
                                  "message": redact(f"{type(exc).__name__}: {exc}")})
            self._append(handle, {"type": "run_finished", "status": "failed",
                                  "duration": round(time.monotonic() - started, 3)})
        finally:
            self._flush_pending(handle)
            unregister_event_listener(listener)
            control.reset(token)
            self._finish(handle)

    def _call_start(self, task_id: str, run_id: str, payload):
        run_fn = self._run_fn or self._default_start
        return run_fn(
            target_image_path=payload.samples[0],
            description=payload.message,
            output_root=str(self.output_root),
            provider=self._provider(),
            task_store=self.task_store,
            task_id=task_id,
            thread_id=run_id,
            target_image_paths=payload.samples,
            target_type=payload.target_type,
        )

    def _default_start(self, **kwargs):
        from core.agent_graph import run_agent_graph

        return run_agent_graph(**kwargs)

    def _call_resume(self, run_id: str, payload):
        resume_fn = self._resume_fn or self._default_resume
        response = {"action": payload.action}
        if payload.feedback:
            response["feedback"] = payload.feedback
        return resume_fn(thread_id=run_id, response=response, provider=self._provider())

    def _default_resume(self, **kwargs):
        from core.agent_graph import resume_agent_graph

        return resume_agent_graph(**kwargs)

    def _provider(self):
        if self._provider_factory is not None:
            return self._provider_factory()
        from providers.vision import build_runtime_provider

        return build_runtime_provider()

    def _review_event(self, interrupt) -> dict:
        value = interrupt.get("value") if isinstance(interrupt, dict) else None
        value = value or {}
        overlay = value.get("overlay_path") or ""
        event = {
            "type": "review_requested",
            "stage": str(value.get("stage") or ""),
            "message": str(value.get("message") or ""),
            "overlay_url": file_url(overlay, self.roots) if overlay else None,
            "best_score": float(value.get("best_score") or 0.0),
        }
        if value.get("stop_reason"):
            event["stop_reason"] = str(value["stop_reason"])
        return event

    # --- 事件缓冲 / 持久化 ---

    def _make_listener(self, handle: RunHandle):
        def listener(core_event: dict) -> None:
            for ui_event in handle.translator.translate(core_event):
                self._ingest(handle, ui_event)
        return listener

    def _append(self, handle: RunHandle, event: dict) -> None:
        self._ingest(handle, event)

    def _ingest(self, handle: RunHandle, event: dict) -> None:
        with handle.condition:
            if event.get("type") in MERGEABLE_TYPES:
                buffered = handle.pending
                if (buffered is not None
                        and buffered.get("type") == event.get("type")
                        and buffered.get("context") == event.get("context")
                        and time.monotonic() - handle.pending_since <= MERGE_WINDOW):
                    buffered["text"] = str(buffered.get("text") or "") + str(event.get("text") or "")
                else:
                    if buffered is not None:
                        self._commit_locked(handle, buffered)
                    handle.pending = event
                    handle.pending_since = time.monotonic()
            else:
                if handle.pending is not None:
                    buffered, handle.pending = handle.pending, None
                    self._commit_locked(handle, buffered)
                self._commit_locked(handle, event)

    def _flush_pending(self, handle: RunHandle) -> None:
        with handle.condition:
            if handle.pending is not None:
                buffered, handle.pending = handle.pending, None
                self._commit_locked(handle, buffered)

    def _commit_locked(self, handle: RunHandle, event: dict) -> None:
        handle.seq += 1
        event["seq"] = handle.seq
        event.setdefault("run_id", handle.run_id)
        event.setdefault("ts", time.time())
        handle.events.append(event)
        self._persist(handle, event)
        handle.condition.notify_all()

    def _persist(self, handle: RunHandle, event: dict) -> None:
        try:
            with handle.stream_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_redact_event(event), ensure_ascii=False,
                                        default=str) + "\n")
        except OSError:
            logger.warning("Could not persist run event to %s", handle.stream_path,
                           exc_info=True)

    def _finish(self, handle: RunHandle) -> None:
        with handle.condition:
            handle.finished = True
            handle.finished_at = time.monotonic()
            handle.condition.notify_all()
        with self._lock:
            if self._active.get(handle.task_id) == handle.run_id:
                del self._active[handle.task_id]

    # --- handle 管理 ---

    def _new_handle_locked(self, task_id: str, run_id: str) -> RunHandle:
        stream_path = self.task_store.task_dir(task_id) / "runs" / run_id / "stream.jsonl"
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        handle = RunHandle(task_id, run_id, stream_path)
        handle.translator = EventTranslator(run_id, self.roots)
        self._handles[(task_id, run_id)] = handle
        return handle

    def _get_or_restore_locked(self, task_id: str, run_id: str) -> RunHandle:
        handle = self._handles.get((task_id, run_id))
        if handle is not None:
            return handle
        stream_path = self.task_store.task_dir(task_id) / "runs" / run_id / "stream.jsonl"
        if not stream_path.is_file():
            raise FileNotFoundError(f"Unknown run: {run_id}")
        handle = RunHandle(task_id, run_id, stream_path)
        handle.translator = EventTranslator(run_id, self.roots)
        events, seq = [], 0
        for line in stream_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue  # 写入一半的行：跳过
            events.append(event)
            seq = max(seq, int(event.get("seq") or 0))
        handle.events = events
        handle.seq = seq
        self._handles[(task_id, run_id)] = handle
        return handle

    @staticmethod
    def _raise_or_missing(run_id: str):
        raise FileNotFoundError(f"Unknown run: {run_id}")

    def _sweep_locked(self) -> None:
        now = time.monotonic()
        for key, handle in list(self._handles.items()):
            if (handle.finished and handle.finished_at is not None
                    and now - handle.finished_at > RETENTION_SECONDS):
                del self._handles[key]

    # --- SSE 流 ---

    def _stream(self, handle: RunHandle, cursor: int):
        while True:
            with handle.condition:
                while True:
                    events = [event for event in handle.events if event["seq"] > cursor]
                    if events:
                        break
                    if handle.finished and handle.pending is None:
                        break
                    if not handle.condition.wait(timeout=SUBSCRIBE_WAIT):
                        events = None
                        break
            if events is None:
                yield ": ping\n\n"
                continue
            if not events:
                break
            for event in events:
                cursor = event["seq"]
                yield sse_frame(event)
        yield "event: end\n\n"
