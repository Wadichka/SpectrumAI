import { create } from "zustand";

export interface HistoryEntry {
  requestId: number;
  timestamp: string;
  status: string;
  inputFilename: string | null;
  topPredictedGroups: string[];
}

interface HistoryStore {
  items: HistoryEntry[];
  total: number;
  setItems: (items: HistoryEntry[], total: number) => void;
  clear: () => void;
}

export const useHistoryStore = create<HistoryStore>((set) => ({
  items: [],
  total: 0,
  setItems: (items, total) => set({ items, total }),
  clear: () => set({ items: [], total: 0 }),
}));
