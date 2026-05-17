import { apiClient } from "@/api/client";
import type { components } from "@/api/types.gen";

export type BatchResponse = components["schemas"]["BatchIdentificationResponse"];
export type BatchItem = components["schemas"]["BatchIdentificationItemResponse"];

export interface BatchOptions {
  topK: number;
  includeGradcam: boolean;
}

export async function postIdentifyBatch(
  files: File[],
  options: BatchOptions,
  signal?: AbortSignal,
): Promise<BatchResponse> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file, file.name);
  }
  formData.append("top_k", String(options.topK));
  formData.append("include_gradcam", String(options.includeGradcam));
  const { data } = await apiClient.post<BatchResponse>(
    "/api/v1/identify/batch",
    formData,
    { signal, headers: { "Content-Type": "multipart/form-data" } },
  );
  return data;
}
