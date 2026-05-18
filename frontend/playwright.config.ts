import { defineConfig, devices } from "@playwright/test";

/**
 * Конфигурация Playwright для e2e SpectrumAI (Этап 16 F, §11.6.3).
 *
 * Тесты запускаются против поднятого через `docker compose up -d` стека:
 * - frontend (nginx) на http://localhost
 * - backend (gunicorn) проксирован через /api/*
 *
 * Запуск: `npm run test:e2e` (headless) или `npm run test:e2e:ui`.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [["html", { outputFolder: "playwright-report", open: "never" }], ["list"]],
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    navigationTimeout: 15_000,
    actionTimeout: 10_000,
    locale: "ru-RU",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
