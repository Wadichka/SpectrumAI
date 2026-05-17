import { apiClient } from "@/api/client";
import type { components } from "@/api/types.gen";

export type SettingsPayload = components["schemas"]["SettingsResponse"];

export async function fetchSettings(): Promise<SettingsPayload> {
  const { data } = await apiClient.get<SettingsPayload>("/api/v1/settings");
  return data;
}

export async function updateSettings(payload: SettingsPayload): Promise<SettingsPayload> {
  const { data } = await apiClient.patch<SettingsPayload>("/api/v1/settings", payload);
  return data;
}
