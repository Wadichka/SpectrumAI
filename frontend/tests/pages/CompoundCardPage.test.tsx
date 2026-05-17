import { render, screen, waitFor } from "@testing-library/react";
import { AxiosError, type AxiosHeaders } from "axios";
import { I18nextProvider } from "react-i18next";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import i18n from "@/i18n";
import CompoundCardPage from "@/pages/CompoundCardPage";
import { useIdentificationStore } from "@/stores/useIdentificationStore";

vi.mock("@/api/compounds", async () => {
  const actual = await vi.importActual<typeof import("@/api/compounds")>("@/api/compounds");
  return {
    ...actual,
    fetchCompoundDetail: vi.fn(),
  };
});

import { fetchCompoundDetail } from "@/api/compounds";

const fetchMock = vi.mocked(fetchCompoundDetail);

function renderPage(id = "100") {
  return render(
    <I18nextProvider i18n={i18n}>
      <MemoryRouter initialEntries={[`/compounds/${id}`]}>
        <Routes>
          <Route path="/compounds/:id" element={<CompoundCardPage />} />
        </Routes>
      </MemoryRouter>
    </I18nextProvider>,
  );
}

const candidateInStore = {
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
};

beforeEach(() => {
  fetchMock.mockReset();
  // Default behaviour: pending promise so stray re-renders during cleanup
  // don't trip on undefined return values from an exhausted *Once mock.
  fetchMock.mockImplementation(() => new Promise(() => {}));
  useIdentificationStore.getState().reset();
});

afterEach(() => {
  useIdentificationStore.getState().reset();
});

describe("CompoundCardPage", () => {
  it("показывает данные из API при успешном запросе", async () => {
    fetchMock.mockResolvedValueOnce({
      id: 100,
      name: "ethanol",
      smiles: "CCO",
      formula: "C2H6O",
      cas_number: "64-17-5",
      iupac_name: null,
      inchi: "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
      inchi_key: "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
      molecular_weight: 46.07,
      functional_groups: ["FG01"],
      spectra_count: 0,
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByText("ethanol")).toBeInTheDocument();
    });
    expect(screen.getByText("C2H6O")).toBeInTheDocument();
    expect(screen.getByText("LFQSCWFLJHTTHZ-UHFFFAOYSA-N")).toBeInTheDocument();
  });

  it("404 + fallback из lastResponse.candidates", async () => {
    const error = new AxiosError("not found", "ERR_BAD_REQUEST");
    error.response = {
      status: 404,
      statusText: "Not Found",
      data: { detail: { code: "ENTITY_NOT_FOUND", message: "not found" } },
      headers: {} as AxiosHeaders,
      config: { headers: {} as AxiosHeaders },
    };
    fetchMock.mockRejectedValueOnce(error);
    useIdentificationStore.getState().setLastResponse({
      request_id: 1,
      predictions: [],
      candidates: [candidateInStore],
      gradcam: null,
      spectrum_length: 3601,
      model_versions: {},
      threshold_mode: "fixed",
      processing_time_ms: 0,
      timestamp: new Date().toISOString(),
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByText("ethanol")).toBeInTheDocument();
    });
    expect(screen.getByText("C2H6O")).toBeInTheDocument();
  });

  it("404 без fallback → Alert «не найдено»", async () => {
    const error = new AxiosError("not found", "ERR_BAD_REQUEST");
    error.response = {
      status: 404,
      statusText: "Not Found",
      data: {},
      headers: {} as AxiosHeaders,
      config: { headers: {} as AxiosHeaders },
    };
    fetchMock.mockRejectedValueOnce(error);
    renderPage("999");
    expect(await screen.findByText(/Соединение не найдено/i)).toBeInTheDocument();
  });
});
