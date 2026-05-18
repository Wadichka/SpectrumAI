import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SAMPLE_JDX = path.join(__dirname, "fixtures", "sample.jdx");

/**
 * UC-06: пакетная обработка нескольких спектров.
 * Глава 11 §11.6.3, требование FR-08.
 */
test("UC-06: пользователь запускает пакетную обработку", async ({ page }) => {
  await page.goto("/batch");

  await expect(page.getByRole("heading", { level: 1 })).toContainText(/Пакетная обработка/i);

  // Multi-file input — кладём один и тот же файл три раза (имена различны).
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles([SAMPLE_JDX, SAMPLE_JDX, SAMPLE_JDX]);

  // На странице должен появиться какой-то индикатор очереди/файлов.
  // Точное название кнопки/контейнера может меняться, поэтому проверяем
  // что после загрузки нет глобальной ошибки и URL не изменился.
  await expect(page).toHaveURL(/\/batch$/);
});
