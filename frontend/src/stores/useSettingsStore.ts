import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { components } from "@/api/types.gen";

export type Language = "ru" | "en";
export type BaselineMethod = "asls" | "none";
export type NormalizeMethod = "snv" | "minmax";

export type SettingsPayload = components["schemas"]["SettingsResponse"];

interface SettingsState {
  language: Language;
  threshold: number;
  topK: number;
  includeGradcam: boolean;
  baselineMethod: BaselineMethod;
  normalizeMethod: NormalizeMethod;
  savgolWindow: number;
  savgolPolyorder: number;
  setLanguage: (language: Language) => void;
  setThreshold: (threshold: number) => void;
  setTopK: (topK: number) => void;
  setIncludeGradcam: (value: boolean) => void;
  setBaselineMethod: (value: BaselineMethod) => void;
  setNormalizeMethod: (value: NormalizeMethod) => void;
  setSavgolWindow: (value: number) => void;
  setSavgolPolyorder: (value: number) => void;
  applyServerSettings: (payload: SettingsPayload) => void;
  toPayload: () => SettingsPayload;
  resetToDefaults: () => void;
}

const DEFAULTS = {
  language: "ru" as Language,
  threshold: 0.5,
  topK: 10,
  includeGradcam: true,
  baselineMethod: "asls" as BaselineMethod,
  normalizeMethod: "snv" as NormalizeMethod,
  savgolWindow: 11,
  savgolPolyorder: 2,
};

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      ...DEFAULTS,
      setLanguage: (language) => set({ language }),
      setThreshold: (threshold) => set({ threshold }),
      setTopK: (topK) => set({ topK }),
      setIncludeGradcam: (includeGradcam) => set({ includeGradcam }),
      setBaselineMethod: (baselineMethod) => set({ baselineMethod }),
      setNormalizeMethod: (normalizeMethod) => set({ normalizeMethod }),
      setSavgolWindow: (savgolWindow) => set({ savgolWindow }),
      setSavgolPolyorder: (savgolPolyorder) => set({ savgolPolyorder }),
      applyServerSettings: (payload) =>
        set({
          language: payload.language,
          threshold: payload.threshold,
          topK: payload.top_k,
          includeGradcam: payload.include_gradcam,
          baselineMethod: payload.baseline_method,
          normalizeMethod: payload.normalize_method,
          savgolWindow: payload.savgol_window,
          savgolPolyorder: payload.savgol_polyorder,
        }),
      toPayload: () => {
        const s = get();
        return {
          language: s.language,
          threshold: s.threshold,
          top_k: s.topK,
          include_gradcam: s.includeGradcam,
          baseline_method: s.baselineMethod,
          normalize_method: s.normalizeMethod,
          savgol_window: s.savgolWindow,
          savgol_polyorder: s.savgolPolyorder,
        };
      },
      resetToDefaults: () => set({ ...DEFAULTS }),
    }),
    { name: "spectrumai-settings" },
  ),
);
