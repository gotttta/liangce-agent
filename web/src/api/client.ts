import type { Health, SampleOut, TaskDetail, TaskSummary } from "./types";

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(code: string, message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {};
  if (init?.body != null && !(init.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, { ...init, headers: { ...headers, ...init?.headers } });
  if (!response.ok) {
    let code = "internal_error";
    let message = `请求失败（HTTP ${response.status}）`;
    try {
      const body = (await response.json()) as { error?: { code: string; message: string } };
      if (body?.error) {
        code = body.error.code;
        message = body.error.message;
      }
    } catch {
      // 非 JSON 错误体，保留默认文案
    }
    throw new ApiError(code, message, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  listTasks: (limit = 50) =>
    request<TaskSummary[]>(`/api/tasks?limit=${encodeURIComponent(limit)}`),
  createTask: (title?: string) =>
    request<TaskDetail>("/api/tasks", {
      method: "POST",
      body: JSON.stringify({ title: title ?? null }),
    }),
  getTask: (taskId: string) => request<TaskDetail>(`/api/tasks/${taskId}`),
  patchTask: (taskId: string, title: string) =>
    request<TaskDetail>(`/api/tasks/${taskId}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),
  deleteTask: (taskId: string) =>
    request<{ ok: boolean }>(`/api/tasks/${taskId}`, { method: "DELETE" }),
  addSamples: (taskId: string, files: File[]) => {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    return request<{ samples: SampleOut[] }>(`/api/tasks/${taskId}/samples`, {
      method: "POST",
      body: form,
    });
  },
  removeSample: (taskId: string, name: string) =>
    request<{ samples: SampleOut[] }>(
      `/api/tasks/${taskId}/samples/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),
};
