import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AxiosError, type AxiosHeaders } from "axios";
import { I18nextProvider } from "react-i18next";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "@/i18n";
import IdentificationPage from "@/pages/IdentificationPage";
import { useIdentificationStore } from "@/stores/useIdentificationStore";
import { useUploadStore } from "@/stores/useUploadStore";

vi.mock("@/api/identify", () => ({
  postIdentify: vi.fn(),
}));

import { postIdentify } from "@/api/identify";

const postIdentifyMock = vi.mocked(postIdentify);

function renderPage() {
  return render(
    <I18nextProvider i18n={i18n}>
      <MemoryRouter initialEntries={["/identify"]}>
        <Routes>
          <Route path="/identify" element={<IdentificationPage />} />
          <Route
            path="/identify/results/:requestId"
            element={<div data-testid="results-page">Результаты</div>}
          />
        </Routes>
      </MemoryRouter>
    </I18nextProvider>,
  );
}

beforeEach(() => {
  postIdentifyMock.mockReset();
  useUploadStore.getState().reset();
  useIdentificationStore.getState().reset();
});

afterEach(() => {
  useUploadStore.getState().reset();
  useIdentificationStore.getState().reset();
});

describe("IdentificationPage", () => {
  it("в idle: drop-zone виден, кнопка Идентифицировать disabled", () => {
    renderPage();
    expect(
      screen.getByRole("button", { name: /Перетащите файл сюда/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Идентифицировать" })).toBeDisabled();
  });

  it("после выбора файла видим preview и активную кнопку", async () => {
    const user = userEvent.setup();
    renderPage();
    const file = new File(["##TITLE=demo"], "demo.jdx", {
      type: "chemical/x-jcamp-dx",
    });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(fileInput, file);
    expect(screen.getByText("demo.jdx")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Идентифицировать" })).toBeEnabled();
  });

  it("отвергает невалидное расширение", async () => {
    renderPage();
    const file = new File(["data"], "image.png", { type: "image/png" });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    // fireEvent.change игнорирует accept-фильтр (в отличие от userEvent.upload),
    // что позволяет проверить серверо-агностичную клиентскую валидацию.
    fireEvent.change(fileInput, { target: { files: [file] } });
    expect(
      await screen.findByText(/Неподдерживаемый формат файла/i),
    ).toBeInTheDocument();
  });

  it("отвергает слишком большой файл", async () => {
    const user = userEvent.setup();
    renderPage();
    const big = new File([new Uint8Array(11 * 1024 * 1024)], "huge.jdx");
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(fileInput, big);
    expect(await screen.findByText(/Файл слишком большой/i)).toBeInTheDocument();
  });

  it("успешный POST → редирект на /identify/results/:id", async () => {
    const user = userEvent.setup();
    postIdentifyMock.mockResolvedValueOnce({
      request_id: 42,
      predictions: [],
      candidates: [],
      gradcam: null,
      spectrum_length: 3601,
      model_versions: { mode: "test" },
      threshold_mode: "fixed",
      processing_time_ms: 12,
      timestamp: new Date().toISOString(),
    });
    renderPage();
    const file = new File(["##TITLE=demo"], "demo.jdx");
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(fileInput, file);
    await user.click(screen.getByRole("button", { name: "Идентифицировать" }));
    await waitFor(() => {
      expect(screen.getByTestId("results-page")).toBeInTheDocument();
    });
  });

  it("ошибка 422 от backend → Alert с сообщением", async () => {
    const user = userEvent.setup();
    const axiosError = new AxiosError(
      "Request failed with status code 422",
      "ERR_BAD_REQUEST",
    );
    axiosError.response = {
      status: 422,
      statusText: "Unprocessable Entity",
      data: { detail: { code: "PARSING_ERROR", message: "broken file" } },
      headers: {} as AxiosHeaders,
      config: { headers: {} as AxiosHeaders },
    };
    postIdentifyMock.mockRejectedValueOnce(axiosError);
    renderPage();
    const file = new File(["##TITLE=demo"], "demo.jdx");
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(fileInput, file);
    await user.click(screen.getByRole("button", { name: "Идентифицировать" }));
    expect(await screen.findByText("broken file")).toBeInTheDocument();
  });
});
