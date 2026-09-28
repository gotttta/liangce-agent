// 与 api/schemas.py 一一对应（计划 §8：前端只认 UI 协议）
export interface SampleOut {
  name: string;
  url: string;
}

export interface RunSummary {
  run_id: string;
  status: string;
  status_label: string;
  started_at: string | null;
  stop_reason: string | null;
}

export type ReviewStage = "reference" | "final" | "retry";

export interface ReviewRequest {
  stage: ReviewStage;
  message: string;
  overlay_url: string | null;
  best_score: number;
  stop_reason: string | null;
}

export interface BestResult {
  score: number;
  pipeline: Record<string, unknown>[];
  notes: string;
}

export interface TaskSummary {
  id: string;
  title: string;
  status: string;
  status_label: string;
  created_at: string;
  updated_at: string;
  running: boolean;
  /** RunManager 里的活动运行；latest_run_id 落盘前（首个节点未中断时）靠它订阅实时流 */
  active_run_id: string | null;
  sample_count: number;
}

export interface TaskDetail extends TaskSummary {
  samples: SampleOut[];
  latest_run_id: string | null;
  runs: RunSummary[];
  pending_review: ReviewRequest | null;
  best: BestResult | null;
}

export interface Health {
  ok: boolean;
  model: string | null;
}

// ---- UI 事件协议（计划 §8，由 api/events.py + api/runs.py 产出）----

export interface UiEventBase {
  seq: number;
  run_id: string;
  ts: number;
}

export interface ImageScore {
  image_id: string;
  iou_mean: number;
  false_positive_count: number;
  false_negative_count: number;
  ref_count: number;
  composite: number;
}

export type RunFinishStatus = "completed" | "awaiting_review" | "failed" | "cancelled";

export type UiEvent =
  | (UiEventBase & {
      type: "run_started";
      message?: string;
      target_type?: string;
      image_count?: number;
      resumed: boolean;
      action?: string;
      feedback?: string;
    })
  | (UiEventBase & { type: "step_started"; step_id: string; node: string; label: string })
  | (UiEventBase & {
      type: "step_finished";
      step_id: string;
      duration: number | null;
      metadata: Record<string, unknown>;
    })
  | (UiEventBase & {
      type: "thinking";
      step_id: string | null;
      context: string | null;
      text: string;
      delta: boolean;
    })
  | (UiEventBase & {
      type: "model_call_started";
      call_id: string;
      model: string;
      message_count: number;
      has_images: boolean;
    })
  | (UiEventBase & {
      type: "model_call_finished";
      call_id: string;
      usage: Record<string, number> | null;
      context_window: number | null;
      duration: number | null;
    })
  | (UiEventBase & { type: "model_output"; call_id: string; text: string })
  | (UiEventBase & {
      type: "tool_started";
      tool_id: string;
      tool: string;
      args: Record<string, unknown>;
    })
  | (UiEventBase & {
      type: "tool_finished";
      tool_id: string;
      tool: string;
      result: unknown;
      success: boolean;
      duration: number | null;
    })
  | (UiEventBase & {
      type: "reference_candidate";
      image_id: string;
      overlay_url: string | null;
      sam_score: number;
      low_quality: boolean;
    })
  | (UiEventBase & {
      type: "iteration_scored";
      iteration: number;
      composite_mean: number;
      best_score_before: number;
      improved: boolean;
      pipeline: Record<string, unknown>[];
      notes: string;
      image_scores: ImageScore[];
    })
  | (UiEventBase & {
      type: "review_requested";
      stage: ReviewStage;
      message: string;
      overlay_url: string | null;
      best_score: number;
      stop_reason?: string | null;
    })
  | (UiEventBase & { type: "error"; message: string; node?: string | null })
  | (UiEventBase & {
      type: "run_finished";
      status: RunFinishStatus;
      best_score?: number;
      duration: number;
    });

export interface RunSnapshot {
  run_id: string;
  task_id: string;
  status: string;
  events: UiEvent[];
}
