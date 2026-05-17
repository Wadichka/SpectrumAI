import type { components } from "@/api/types.gen";

type BatchItem = components["schemas"]["BatchIdentificationItemResponse"];

const HEADER = [
  "filename",
  "status",
  "request_id",
  "top1_compound_id",
  "top1_name",
  "top1_smiles",
  "top1_score",
  "candidates_count",
  "processing_time_ms",
  "error_code",
  "error_message",
] as const;

function escapeCell(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return "";
  const raw = String(value);
  if (/[",\n;]/.test(raw)) return `"${raw.replace(/"/g, '""')}"`;
  return raw;
}

export function serializeBatchToCsv(items: readonly BatchItem[]): string {
  const lines: string[] = [HEADER.join(",")];
  for (const item of items) {
    const result = item.result ?? null;
    const top1 = result?.candidates?.[0] ?? null;
    const row = [
      escapeCell(item.filename),
      escapeCell(item.status),
      escapeCell(result?.request_id ?? null),
      escapeCell(top1?.compound_id ?? null),
      escapeCell(top1?.name ?? null),
      escapeCell(top1?.smiles ?? null),
      escapeCell(top1?.score != null ? top1.score.toFixed(4) : null),
      escapeCell(result?.candidates?.length ?? 0),
      escapeCell(result?.processing_time_ms ?? null),
      escapeCell(item.error?.code ?? null),
      escapeCell(item.error?.message ?? null),
    ];
    lines.push(row.join(","));
  }
  return lines.join("\n");
}
