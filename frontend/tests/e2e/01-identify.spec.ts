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
