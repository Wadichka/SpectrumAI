import { create } from "zustand";

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
  errorMessage: string | null;
  setState: (state: IdentificationState) => void;
  setLastRequestId: (id: number | null) => void;
  setError: (message: string | null) => void;
  reset: () => void;
}

export const useIdentificationStore = create<IdentificationStore>((set) => ({
  state: "idle",
  lastRequestId: null,
  errorMessage: null,
  setState: (state) => set({ state }),
  setLastRequestId: (lastRequestId) => set({ lastRequestId }),
  setError: (errorMessage) => set({ errorMessage, state: errorMessage ? "error" : "idle" }),
  reset: () => set({ state: "idle", lastRequestId: null, errorMessage: null }),
}));
