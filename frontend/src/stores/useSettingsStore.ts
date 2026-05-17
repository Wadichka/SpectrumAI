import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Language = "ru" | "en";

interface SettingsState {
  language: Language;
  threshold: number;
  topK: number;
  setLanguage: (language: Language) => void;
  setThreshold: (threshold: number) => void;
  setTopK: (topK: number) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      language: "ru",
      threshold: 0.5,
      topK: 10,
      setLanguage: (language) => set({ language }),
      setThreshold: (threshold) => set({ threshold }),
      setTopK: (topK) => set({ topK }),
    }),
    { name: "spectrumai-settings" },
  ),
);
