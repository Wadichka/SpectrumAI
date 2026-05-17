import { create } from "zustand";

import type { IdentificationResponse } from "@/api/identify";

export type IdentificationState =
  | "idle"
  | "preprocessing"
  | "analyzing"
  | "searching"
  | "results"
  | "error";

interface IdentificationStore {
  state: IdentificationState;
  lastRequestId: number | null;
  lastResponse: IdentificationResponse | null;
  errorMessage: string | null;
  setState: (state: IdentificationState) => void;
  setLastRequestId: (id: number | null) => void;
  setLastResponse: (response: IdentificationResponse | null) => void;
  setError: (message: string | null) => void;
  reset: () => void;
}

export const useIdentificationStore = create<IdentificationStore>((set) => ({
  state: "idle",
  lastRequestId: null,
  lastResponse: null,
  errorMessage: null,
  setState: (state) => set({ state }),
  setLastRequestId: (lastRequestId) => set({ lastRequestId }),
  setLastResponse: (lastResponse) => set({ lastResponse }),
  setError: (errorMessage) => set({ errorMessage, state: errorMessage ? "error" : "idle" }),
  reset: () =>
    set({
      state: "idle",
      lastRequestId: null,
      lastResponse: null,
      errorMessage: null,
    }),
}));
