import { create } from "zustand";

import type { BatchResponse } from "@/api/batch";

export type BatchStatus = "idle" | "processing" | "done" | "error";

interface BatchState {
  files: File[];
  status: BatchStatus;
  response: BatchResponse | null;
  errorMessage: string | null;
  addFiles: (files: File[]) => void;
  removeFile: (index: number) => void;
  clearFiles: () => void;
  setProcessing: () => void;
  setResponse: (response: BatchResponse) => void;
  setError: (message: string) => void;
  reset: () => void;
}

export const useBatchStore = create<BatchState>((set) => ({
  files: [],
  status: "idle",
  response: null,
  errorMessage: null,
  addFiles: (newFiles) =>
    set((state) => ({
      files: [...state.files, ...newFiles],
      status: "idle",
      errorMessage: null,
    })),
  removeFile: (index) =>
    set((state) => ({
      files: state.files.filter((_, i) => i !== index),
    })),
  clearFiles: () =>
    set({ files: [], status: "idle", response: null, errorMessage: null }),
  setProcessing: () => set({ status: "processing", errorMessage: null }),
  setResponse: (response) => set({ status: "done", response, errorMessage: null }),
  setError: (message) => set({ status: "error", errorMessage: message }),
  reset: () =>
    set({ files: [], status: "idle", response: null, errorMessage: null }),
}));
