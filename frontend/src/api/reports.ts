import { apiClient } from "@/api/client";
import type { IdentificationResponse } from "@/api/identify";

/**
 * Тянет PDF-отчёт с бэкенда: stateless POST, payload — последний
 * IdentificationResponse из стора. Возвращает Blob готовый для скачивания.
 */
export async function postIdentificationReport(
  response: IdentificationResponse,
): Promise<Blob> {
  const { data } = await apiClient.post<Blob>(
    "/api/v1/reports/identification",
    response,
    { responseType: "blob" },
  );
  return data;
}
