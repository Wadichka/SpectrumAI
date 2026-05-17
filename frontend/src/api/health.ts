import { apiClient } from "@/api/client";
import type { components } from "@/api/types.gen";

export type HealthResponse = components["schemas"]["HealthResponse"];

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>("/api/v1/health");
  return data;
}
