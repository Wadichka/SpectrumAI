import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { I18nextProvider } from "react-i18next";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "@/i18n";
import HistoryPage from "@/pages/HistoryPage";

vi.mock("@/api/history", () => ({
  fetchHistory: vi.fn(),
}));

import { fetchHistory } from "@/api/history";

const fetchMock = vi.mocked(fetchHistory);

function renderPage() {
  return render(
    <I18nextProvider i18n={i18n}>
      <MemoryRouter initialEntries={["/history"]}>
        <Routes>
          <Route path="/history" element={<HistoryPage />} />
          <Route
            path="/identify/results/:requestId"
            element={<div data-testid="results-page" />}
          />
          <Route path="/identify" element={<div data-testid="identify-page" />} />
        </Routes>
      </MemoryRouter>
    </I18nextProvider>,
  );
}

beforeEach(() => {
  fetchMock.mockReset();
  void i18n.changeLanguage("ru");
});

describe("HistoryPage", () => {
  it("показывает empty-state, когда сервер вернул пустой список", async () => {
    fetchMock.mockResolvedValueOnce({ data: [], page: 1, size: 20, total: 0 });
    renderPage();
    await waitFor(() => {
      expect(screen.getByText(/История пуста/i)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /К идентификации/i })).toBeInTheDocument();
  });

  it("рендерит таблицу со строками истории", async () => {
    fetchMock.mockResolvedValueOnce({
      data: [
        {
          request_id: 7,
          timestamp: new Date("2026-05-01T10:30:00Z").toISOString(),
          status: "success",
          processing_time_ms: 123,
          input_filename: "example.jdx",
          top_predicted_groups: ["alcohol_OH"],
        },
      ],
      page: 1,
      size: 20,
      total: 1,
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByText("example.jdx")).toBeInTheDocument();
    });
    expect(screen.getByText("alcohol_OH")).toBeInTheDocument();
    expect(screen.getByText(/123 ms/)).toBeInTheDocument();
  });

  it("клик на «Открыть» ведёт на страницу результатов", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce({
      data: [
        {
          request_id: 42,
          timestamp: new Date("2026-05-01T10:30:00Z").toISOString(),
          status: "success",
          processing_time_ms: 50,
          input_filename: "f.jdx",
          top_predicted_groups: [],
        },
      ],
      page: 1,
      size: 20,
      total: 1,
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByText("f.jdx")).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /Открыть/i }));
    expect(screen.getByTestId("results-page")).toBeInTheDocument();
  });
});
