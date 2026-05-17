import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { I18nextProvider } from "react-i18next";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "@/i18n";
import BatchPage from "@/pages/BatchPage";
import { useBatchStore } from "@/stores/useBatchStore";

vi.mock("@/api/batch", () => ({
  postIdentifyBatch: vi.fn(),
}));

import { postIdentifyBatch } from "@/api/batch";

const postMock = vi.mocked(postIdentifyBatch);

function renderPage() {
  return render(
    <I18nextProvider i18n={i18n}>
      <MemoryRouter initialEntries={["/batch"]}>
        <BatchPage />
      </MemoryRouter>
    </I18nextProvider>,
  );
}

function makeFile(name: string, size = 16): File {
  return new File([new Uint8Array(size)], name, { type: "chemical/x-jcamp-dx" });
}

beforeEach(() => {
  postMock.mockReset();
  postMock.mockImplementation(() => new Promise(() => {}));
  useBatchStore.getState().reset();
  void i18n.changeLanguage("ru");
});

describe("BatchPage", () => {
  it("показывает empty-state и disabled-кнопку при отсутствии файлов", () => {
    renderPage();
    expect(screen.getByText(/Файлы ещё не выбраны/i)).toBeInTheDocument();
    const runButton = screen.getByRole("button", { name: /Запустить обработку/i });
    expect(runButton).toBeDisabled();
  });

  it("после выбора файлов рендерит таблицу и активирует кнопку", async () => {
    renderPage();
    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    expect(fileInput).not.toBeNull();
    fireEvent.change(fileInput, {
      target: {
        files: [makeFile("a.jdx"), makeFile("b.jdx")],
      },
    });
    await waitFor(() => {
      expect(screen.getByText("a.jdx")).toBeInTheDocument();
    });
    expect(screen.getByText("b.jdx")).toBeInTheDocument();
    const runButton = screen.getByRole("button", { name: /Запустить обработку/i });
    expect(runButton).not.toBeDisabled();
  });

  it("submit → таблица результатов с status-бейджами", async () => {
    const user = userEvent.setup();
    postMock.mockResolvedValueOnce({
      items: [
        {
          filename: "ok.jdx",
          status: "success",
          result: {
            request_id: 1,
            predictions: [],
            candidates: [
              {
                rank: 1,
                compound_id: 100,
                smiles: "CCO",
                name: "ethanol",
                formula: "C2H6O",
                cas_number: null,
                score: 0.95,
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
      ],
      total_processing_time_ms: 80,
    });
    renderPage();
    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    fireEvent.change(fileInput, {
      target: { files: [makeFile("ok.jdx"), makeFile("bad.jdx")] },
    });
    await waitFor(() => {
      expect(screen.getByText("ok.jdx")).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /Запустить обработку/i }));
    await waitFor(() => {
      expect(screen.getByText("ethanol")).toBeInTheDocument();
    });
    expect(screen.getByText(/успех/i)).toBeInTheDocument();
    expect(screen.getByText(/ошибка/i)).toBeInTheDocument();
  });

  it("Экспорт CSV вызывает URL.createObjectURL с Blob", async () => {
    const user = userEvent.setup();
    postMock.mockResolvedValueOnce({
      items: [
        {
          filename: "ok.jdx",
          status: "success",
          result: {
            request_id: 1,
            predictions: [],
            candidates: [],
            gradcam: null,
            spectrum: null,
            spectrum_length: 3601,
            model_versions: {},
            threshold_mode: "fixed",
            processing_time_ms: 12,
            timestamp: new Date().toISOString(),
          },
          error: null,
        },
      ],
      total_processing_time_ms: 12,
    });
    if (typeof URL.createObjectURL !== "function") {
      URL.createObjectURL = () => "blob:test";
    }
    if (typeof URL.revokeObjectURL !== "function") {
      URL.revokeObjectURL = () => {};
    }
    const createSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:test");
    const revokeSpy = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    renderPage();
    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [makeFile("ok.jdx")] } });
    await waitFor(() => expect(screen.getByText("ok.jdx")).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: /Запустить обработку/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /Экспорт CSV/i })).toBeInTheDocument(),
    );
    await user.click(screen.getByRole("button", { name: /Экспорт CSV/i }));
    expect(createSpy).toHaveBeenCalledOnce();
    expect(createSpy.mock.calls[0][0]).toBeInstanceOf(Blob);
    createSpy.mockRestore();
    revokeSpy.mockRestore();
  });
});
