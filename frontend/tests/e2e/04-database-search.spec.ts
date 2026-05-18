import { expect, test } from "@playwright/test";

/**
 * UC-04: поиск соединений в локальной базе.
 * Глава 11 §11.6.3, требование FR-02/FR-03.
 *
 * На фазе 1 база может быть пустой (синтетика без сидов БД) — тогда
 * страница показывает "База пуста". Тест проверяет, что навигация
 * стабильно работает и search-input доступен.
 */
test("UC-04: пользователь открывает базу соединений", async ({ page }) => {
  await page.goto("/compounds");

  await expect(page.getByRole("heading", { level: 1 })).toContainText(/База соединений/i);

  // CompoundsFilters рендерит текстовый Input с aria-label из i18n.
  const search = page.getByRole("textbox").first();
  await expect(search).toBeVisible({ timeout: 10_000 });

  // Проверяем, что ввод работает.
  await search.fill("ethanol");
  await expect(search).toHaveValue("ethanol");
});
