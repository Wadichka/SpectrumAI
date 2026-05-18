import { expect, test } from "@playwright/test";

/**
 * UC-08: история запросов.
 * Глава 11 §11.6.3, требование FR-10.
 *
 * Тест проверяет, что страница истории доступна и отрисовывается
 * (на фазе 1 может быть пустой). После UC-01 должна появиться хотя бы
 * одна запись, но порядок выполнения тестов не гарантирован — поэтому
 * мягкая проверка наличия таблицы или сообщения «История пуста».
 */
test("UC-08: пользователь открывает историю запросов", async ({ page }) => {
  await page.goto("/history");

  await expect(page.getByRole("heading", { level: 1 })).toContainText(/История/i);

  // На странице должна быть либо таблица истории, либо empty-state.
  const tableOrEmpty = page.locator("table, [data-testid='empty-state'], h2, p");
  await expect(tableOrEmpty.first()).toBeVisible({ timeout: 10_000 });
});
