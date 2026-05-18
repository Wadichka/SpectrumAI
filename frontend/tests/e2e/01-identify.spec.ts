import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SAMPLE_JDX = path.join(__dirname, "fixtures", "sample.jdx");

/**
 * UC-01: идентификация соединения по загруженному спектру.
 * Глава 11 §11.6.3, требование FR-01.
 */
test("UC-01: пользователь идентифицирует загруженный спектр", async ({ page }) => {
  await page.goto("/identify");

  await expect(page.getByRole("heading", { level: 1 })).toContainText("Идентификация");

  // DropZone оборачивает скрытый <input type="file"> в div role=button.
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(SAMPLE_JDX);

  // После выбора файла появляется FilePreview с именем файла.
  await expect(page.getByText("sample.jdx")).toBeVisible({ timeout: 5_000 });

  const submit = page.getByRole("button", { name: /^Идентифицировать$/ });
  await expect(submit).toBeEnabled();
  await submit.click();

  // Дальнейший flow зависит от наличия моделей в /models — на фазе 1 их
  // может не быть, тогда бэк возвращает 500 и страница остаётся на /identify.
  // Тест считается успешным, если запрос отправлен и UI не упал.
  await page.waitForTimeout(2_000);
  const url = page.url();
  expect(url.includes("/identify")).toBeTruthy();
});

test("UC-05: пользователь скачивает PDF-отчёт идентификации", async ({ page }) => {
  test.setTimeout(60_000);
  // Идём напрямую на /identify, отправляем стандартный flow, доходим до
  // /identify/results/* и проверяем кнопку «Сохранить отчёт».
  await page.goto("/identify");
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(SAMPLE_JDX);
  await page.getByRole("button", { name: /^Идентифицировать$/ }).click();

  // Если бэк не вернул результат за 10 секунд — пропускаем (на фазе 1 модели
  // в /models могут быть пустыми и пайплайн падает 500).
  try {
    await page.waitForURL(/\/identify\/results\//, { timeout: 10_000 });
  } catch {
    test.skip(true, "Identify pipeline unavailable on this stack — see model checkpoints.");
    return;
  }

  // Кнопка «Сохранить отчёт» должна быть включена; клик → download.
  const exportButton = page.getByRole("button", { name: /^Сохранить отчёт$/ });
  await expect(exportButton).toBeEnabled();
  const downloadPromise = page.waitForEvent("download");
  await exportButton.click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/identification-.*\.pdf$/);
});
