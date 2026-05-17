import { apiClient } from "@/api/client";
import type { components } from "@/api/types.gen";

export type CompoundDetail = components["schemas"]["CompoundDetailResponse"];
export type CompoundSummary = components["schemas"]["CompoundSummary"];
export type PaginatedCompounds = components["schemas"]["PaginatedCompoundsResponse"];

export interface CompoundsQuery {
  q?: string;
  functional_groups?: string[];
  page: number;
  size: number;
}

export async function fetchCompoundDetail(id: number): Promise<CompoundDetail> {
  const { data } = await apiClient.get<CompoundDetail>(`/api/v1/compounds/${id}`);
  return data;
}

export async function fetchCompounds(query: CompoundsQuery): Promise<PaginatedCompounds> {
  const params = new URLSearchParams();
  params.set("page", String(query.page));
  params.set("size", String(query.size));
  if (query.q) params.set("q", query.q);
  if (query.functional_groups) {
    for (const code of query.functional_groups) params.append("functional_groups", code);
  }
  const { data } = await apiClient.get<PaginatedCompounds>(
    `/api/v1/compounds?${params.toString()}`,
  );
  return data;
}

export function structureSvgUrlBySmiles(
  smiles: string,
  opts: { width?: number; height?: number } = {},
): string {
  const params = new URLSearchParams({ smiles });
  if (opts.width) params.set("width", String(opts.width));
  if (opts.height) params.set("height", String(opts.height));
  const base = apiClient.defaults.baseURL ?? "";
  return `${base}/api/v1/compounds/structure.svg?${params.toString()}`;
}
