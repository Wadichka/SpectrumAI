import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { I18nextProvider } from "react-i18next";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "@/i18n";
import ResultsPage from "@/pages/ResultsPage";
import { useIdentificationStore } from "@/stores/useIdentificationStore";

vi.mock("react-plotly.js", () => ({
  default: () => <div data-testid="plotly-chart" />,
}));

vi.mock("@/api/reports", () => ({
  postIdentificationReport: vi.fn(async () => new Blob(["%PDF-fake"], { type: "application/pdf" })),
}));

function renderPage() {
  return render(
    <I18nextProvider i18n={i18n}>
      <MemoryRouter initialEntries={["/identify/results/42"]}>
        <Routes>
          <Route path="/identify" element={<div data-testid="identify-page">/identify</div>} />
          <Route path="/identify/results/:requestId" element={<ResultsPage />} />
        </Routes>
      </MemoryRouter>
    </I18nextProvider>,
  );
}

const sampleResponse = {
  request_id: 42,
  predictions: [
    {
      code: "FG01",
      name: "alcohol_OH",
      probability: 0.92,
      threshold: 0.5,
      predicted: true,
    },
    {
      code: "FG02",
      name: "phenol_OH",
      probability: 0.1,
      threshold: 0.5,
      predicted: false,
    },
  ],
  candidates: [
    {
      rank: 1,
      compound_id: 100,
      smiles: "CCO",
      name: "ethanol",
      formula: "C2H6O",
      cas_number: "64-17-5",
      score: 0.95,
      consistent: true,
      jaccard: 1.0,
      matched_groups: ["alcohol_OH"],
      missing_groups: [],
      extra_groups: [],
    },
  ],
  gradcam: {
    group_code: "FG01",
    group_name: "alcohol_OH",
    values: new Array(8).fill(0.5),
  },
  spectrum: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  spectrum_length: 8,
  model_versions: { mode: "contrastive", checkpoint: "best.pt" },
  threshold_mode: "fixed",
  processing_time_ms: 25,
  timestamp: new Date().toISOString(),
};

beforeEach(() => {
  useIdentificationStore.getState().reset();
});

afterEach(() => {
  useIdentificationStore.getState().reset();
});

describe("ResultsPage", () => {
  it("показывает предупреждение при отсутствии lastResponse", () => {
    renderPage();
    expect(screen.getByText(/Результаты идентификации потеряны/i)).toBeInTheDocument();
  });

  it("с lastResponse рисует Plotly, бейджи и кандидата", () => {
    useIdentificationStore.getState().setLastResponse(sampleResponse);
    renderPage();
    expect(screen.getByTestId("plotly-chart")).toBeInTheDocument();
    expect(screen.getAllByText(/alcohol_OH/i).length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: /ethanol/i })).toBeInTheDocument();
    expect(screen.getByText(/score 0.950/)).toBeInTheDocument();
  });

  it("кнопка Новый поиск ведёт на /identify", async () => {
    const user = userEvent.setup();
    useIdentificationStore.getState().setLastResponse(sampleResponse);
    renderPage();
    await user.click(screen.getByRole("button", { name: /Новый поиск/i }));
    expect(screen.getByTestId("identify-page")).toBeInTheDocument();
  });

  it("кнопка Сохранить отчёт вызывает API и инициирует загрузку blob", async () => {
    const user = userEvent.setup();
    const { postIdentificationReport } = await import("@/api/reports");
    const reportSpy = vi.mocked(postIdentificationReport);
    reportSpy.mockClear();
    const createObjectURL = vi.fn(() => "blob:fake-url");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL });

    useIdentificationStore.getState().setLastResponse(sampleResponse);
    renderPage();
    await user.click(screen.getByRole("button", { name: /Сохранить отчёт/i }));

    expect(reportSpy).toHaveBeenCalledWith(sampleResponse);
    expect(createObjectURL).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalled();

    vi.unstubAllGlobals();
  });
});
