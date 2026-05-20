import axios from "axios";

const baseURL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const apiClient = axios.create({
  baseURL,
  timeout: 30_000,
  withCredentials: false,
  // Защитный пояс от агрессивных HTTP-кэшей (особенно после ребилда фронта,
  // когда браузер тянет старый JS-бандл и пытается переиспользовать ответы).
  headers: {
    "Cache-Control": "no-cache",
    Pragma: "no-cache",
  },
});
