import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { I18nextProvider } from "react-i18next";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "@/i18n";
import DatabasePage from "@/pages/DatabasePage";

vi.mock("@/api/compounds", async () => {
  const actual = await vi.importActual<typeof import("@/api/compounds")>("@/api/compounds");
  return { ...actual, fetchCompounds: vi.fn() };
});
vi.mock("@/api/functionalGroups", () => ({
  fetchFunctionalGroups: vi.fn(),
}));

import { fetchCompounds } from "@/api/compounds";
import { fetchFunctionalGroups } from "@/api/functionalGroups";

const compoundsMock = vi.mocked(fetchCompounds);
const groupsMock = vi.mocked(fetchFunctionalGroups);

function renderPage() {
  return render(
    <I18nextProvider i18n={i18n}>
      <MemoryRouter initialEntries={["/compounds"]}>
        <DatabasePage />
      </MemoryRouter>
    </I18nextProvider>,
  );
}

beforeEach(() => {
  compoundsMock.mockReset();
  groupsMock.mockReset();
  groupsMock.mockResolvedValue([
    { code: "FG01", name: "alcohol_OH", description: null, characteristic_bands: null },
    { code: "FG02", name: "phenol_OH", description: null, characteristic_bands: null },
  ]);
  void i18n.changeLanguage("ru");
});

describe("DatabasePage", () => {
  it("показывает empty-state при пустом ответе", async () => {
    compoundsMock.mockResolvedValueOnce({ data: [], page: 1, size: 20, total: 0 });
    renderPage();
    await waitFor(() => {
      expect(screen.getByText(/Совпадений не найдено/i)).toBeInTheDocument();
    });
  });

  it("рендерит строки с соединениями", async () => {
    compoundsMock.mockResolvedValueOnce({
      data: [
        {
          id: 1,
          name: "ethanol",
          smiles: "CCO",
          formula: "C2H6O",
          cas_number: "64-17-5",
        },
      ],
      page: 1,
      size: 20,
      total: 1,
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByText("ethanol")).toBeInTheDocument();
    });
    expect(screen.getByText("CCO")).toBeInTheDocument();
    expect(screen.getByText("C2H6O")).toBeInTheDocument();
  });

  it("клик по FG-чипу добавляет код в запрос", async () => {
    const user = userEvent.setup();
    compoundsMock.mockResolvedValue({ data: [], page: 1, size: 20, total: 0 });
    renderPage();
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /FG01/i })).toBeInTheDocument();
    });
    compoundsMock.mockClear();
    await user.click(screen.getByRole("button", { name: /FG01/i }));
    await waitFor(() => {
      expect(compoundsMock).toHaveBeenCalledWith(
        expect.objectContaining({ functional_groups: ["FG01"] }),
      );
    });
  });
});
