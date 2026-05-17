import { apiClient } from "@/api/client";
import type { components } from "@/api/types.gen";

export type IdentificationResponse = components["schemas"]["IdentificationResponse"];
export type ApiError = components["schemas"]["ApiError"];

export interface IdentifyOptions {
  includeGradcam: boolean;
  topK: number;
}

export async function postIdentify(
  file: File,
  options: IdentifyOptions,
  signal?: AbortSignal,
): Promise<IdentificationResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("include_gradcam", String(options.includeGradcam));
  form.append("top_k", String(options.topK));
  const { data } = await apiClient.post<IdentificationResponse>("/api/v1/identify", form, {
    signal,
  });
  return data;
}
