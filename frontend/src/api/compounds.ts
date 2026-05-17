import { apiClient } from "@/api/client";
import type { components } from "@/api/types.gen";

export type CompoundDetail = components["schemas"]["CompoundDetailResponse"];

export async function fetchCompoundDetail(id: number): Promise<CompoundDetail> {
  const { data } = await apiClient.get<CompoundDetail>(`/api/v1/compounds/${id}`);
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
