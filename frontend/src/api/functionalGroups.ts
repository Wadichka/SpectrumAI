import { apiClient } from "@/api/client";
import type { components } from "@/api/types.gen";

export type FunctionalGroup = components["schemas"]["FunctionalGroupResponse"];

export async function fetchFunctionalGroups(): Promise<FunctionalGroup[]> {
  const { data } = await apiClient.get<FunctionalGroup[]>("/api/v1/functional-groups");
  return data;
}
