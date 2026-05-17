import { apiClient } from "@/api/client";
import type { components } from "@/api/types.gen";

export type HistoryEntry = components["schemas"]["HistoryEntryResponse"];
export type PaginatedHistory = components["schemas"]["PaginatedHistoryResponse"];

export interface HistoryQuery {
  page: number;
  size: number;
  date_from?: string;
  date_to?: string;
  status?: string;
}

export async function fetchHistory(query: HistoryQuery): Promise<PaginatedHistory> {
  const params: Record<string, string | number> = { page: query.page, size: query.size };
  if (query.date_from) params.date_from = query.date_from;
  if (query.date_to) params.date_to = query.date_to;
  if (query.status && query.status !== "all") params.status = query.status;
  const { data } = await apiClient.get<PaginatedHistory>("/api/v1/history", { params });
  return data;
}
