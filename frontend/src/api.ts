import type {
  AttentionResult,
  EvaluationReport,
  GenerationResult,
  HealthStatus,
  ModelMetadata,
} from "./types";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let message = "The request could not be completed.";
    try {
      const payload = (await response.json()) as { detail?: string | Array<{ msg: string }> };
      if (typeof payload.detail === "string") message = payload.detail;
      if (Array.isArray(payload.detail)) message = payload.detail.map((item) => item.msg).join(" ");
    } catch {
      // Keep the plain fallback when the server does not return JSON.
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthStatus>("/api/v1/health"),
  model: () => request<ModelMetadata>("/api/v1/model"),
  evaluation: () => request<EvaluationReport>("/api/v1/evaluation"),
  generate: (payload: { prompt: string; max_new_tokens: number; temperature: number }) =>
    request<GenerationResult>("/api/v1/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  attention: (payload: { prompt: string; layer: number; head: number | null }) =>
    request<AttentionResult>("/api/v1/attention", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
