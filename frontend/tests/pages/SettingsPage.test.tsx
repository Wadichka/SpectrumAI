import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { I18nextProvider } from "react-i18next";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "@/i18n";
import SettingsPage from "@/pages/SettingsPage";
import { useSettingsStore } from "@/stores/useSettingsStore";

vi.mock("@/api/settings", () => ({
  fetchSettings: vi.fn(),
  updateSettings: vi.fn(),
}));

import { fetchSettings, updateSettings } from "@/api/settings";

const fetchMock = vi.mocked(fetchSettings);
const updateMock = vi.mocked(updateSettings);

const defaultPayload = {
  language: "ru" as const,
  include_gradcam: true,
  top_k: 10,
  threshold: 0.5,
  baseline_method: "asls" as const,
  normalize_method: "snv" as const,
  savgol_window: 11,
  savgol_polyorder: 2,
};

function renderPage() {
  return render(
    <I18nextProvider i18n={i18n}>
      <MemoryRouter initialEntries={["/settings"]}>
        <SettingsPage />
      </MemoryRouter>
    </I18nextProvider>,
  );
}

beforeEach(() => {
  fetchMock.mockReset();
  updateMock.mockReset();
  fetchMock.mockImplementation(() => new Promise(() => {}));
  updateMock.mockImplementation(() => new Promise(() => {}));
  useSettingsStore.getState().resetToDefaults();
  void i18n.changeLanguage("ru");
});

describe("SettingsPage", () => {
  it("применяет дефолты, полученные от GET /settings", async () => {
    fetchMock.mockResolvedValueOnce({ ...defaultPayload, top_k: 25, threshold: 0.7 });
    renderPage();
    await waitFor(() => {
      expect(useSettingsStore.getState().topK).toBe(25);
    });
    expect(useSettingsStore.getState().threshold).toBeCloseTo(0.7);
  });

  it("Save вызывает updateSettings и показывает success-alert", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(defaultPayload);
    updateMock.mockImplementation(async (p) => p);
    renderPage();
    await waitFor(() => {
      expect(useSettingsStore.getState().topK).toBe(10);
    });
    await user.click(screen.getByRole("button", { name: /Сохранить/i }));
    await waitFor(() => {
      expect(updateMock).toHaveBeenCalledOnce();
    });
    expect(screen.getByText(/Настройки сохранены/i)).toBeInTheDocument();
  });

  it("Reset возвращает значения по умолчанию", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce({ ...defaultPayload, top_k: 30 });
    renderPage();
    await waitFor(() => {
      expect(useSettingsStore.getState().topK).toBe(30);
    });
    await user.click(screen.getByRole("button", { name: /Сбросить к умолчаниям/i }));
    expect(useSettingsStore.getState().topK).toBe(10);
    expect(useSettingsStore.getState().threshold).toBeCloseTo(0.5);
  });
});
