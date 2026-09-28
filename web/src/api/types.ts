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
