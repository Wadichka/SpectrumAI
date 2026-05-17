import { create } from "zustand";

interface UploadState {
  file: File | null;
  setFile: (file: File | null) => void;
  reset: () => void;
}

export const useUploadStore = create<UploadState>((set) => ({
  file: null,
  setFile: (file) => set({ file }),
  reset: () => set({ file: null }),
}));
