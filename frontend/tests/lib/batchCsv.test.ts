import { describe, expect, it } from "vitest";

import { serializeBatchToCsv } from "@/lib/batchCsv";

describe("serializeBatchToCsv", () => {
  it("пустой items → только header", () => {
    const csv = serializeBatchToCsv([]);
    expect(csv.split("\n")).toHaveLength(1);
    expect(csv).toMatch(/^filename,status,/);
  });

  it("строка с success + ошибкой собирается с правильными колонками", () => {
    const csv = serializeBatchToCsv([
      {
        filename: "ok.jdx",
        status: "success",
        result: {
          request_id: 7,
          predictions: [],
          candidates: [
            {
              rank: 1,
              compound_id: 100,
              smiles: "CCO",
              name: "ethanol",
              formula: "C2H6O",
              cas_number: "64-17-5",
              score: 0.9512,
              consistent: true,
              jaccard: 1,
              matched_groups: ["alcohol_OH"],
              missing_groups: [],
              extra_groups: [],
            },
          ],
          gradcam: null,
          spectrum: null,
          spectrum_length: 3601,
          model_versions: {},
          threshold_mode: "fixed",
          processing_time_ms: 50,
          timestamp: new Date().toISOString(),
        },
        error: null,
      },
      {
        filename: "bad.jdx",
        status: "error",
        result: null,
        error: { code: "PARSING_FAILED", message: "broken", details: null },
      },
    ]);
    const lines = csv.split("\n");
    expect(lines).toHaveLength(3);
    expect(lines[1]).toContain("ok.jdx");
    expect(lines[1]).toContain("ethanol");
    expect(lines[1]).toContain("0.9512");
    expect(lines[2]).toContain("PARSING_FAILED");
    expect(lines[2]).toContain("bad.jdx");
  });

  it("экранирует запятые и кавычки", () => {
    const csv = serializeBatchToCsv([
      {
        filename: 'has,comma"and"quote.csv',
        status: "error",
        result: null,
        error: { code: "X", message: "line1\nline2", details: null },
      },
    ]);
    expect(csv).toContain('"has,comma""and""quote.csv"');
    expect(csv).toContain('"line1\nline2"');
  });
});
