# Инициализация фронтенда SpectrumAI

> На этапе 0 инфраструктура подготовлена, но сам React-проект ещё не создан —
> по требованию автора шаги `npm` запускаются вручную. После выполнения шагов
> ниже в каталоге появятся `package.json`, `vite.config.ts`, `src/` и пр.

Стек зафиксирован в `CLAUDE.md`, §4 (фронтенд). Менять без согласования нельзя.

## 1. Создать проект через Vite

Из корня репозитория:

```bash
# Каталог frontend/ уже существует и содержит Dockerfile, SETUP.md, .gitkeep.
# Vite не любит непустую целевую папку, поэтому временно убираем артефакты:
mv frontend/Dockerfile frontend/SETUP.md frontend/.gitkeep /tmp/

# Создаём проект Vite + React + TypeScript (Vite спросит подтверждение):
npm create vite@latest frontend -- --template react-ts

# Возвращаем артефакты на место:
mv /tmp/Dockerfile /tmp/SETUP.md /tmp/.gitkeep frontend/

cd frontend
npm install
```

В Windows PowerShell вместо `/tmp/` используй `$env:TEMP`.

## 2. Установить runtime-зависимости (CLAUDE.md §4)

```bash
npm install \
  react-router-dom \
  axios \
  pinia \
  plotly.js react-plotly.js \
  lucide-react \
  i18next react-i18next
```

## 3. Установить dev-зависимости

```bash
npm install -D \
  vitest \
  @testing-library/react \
  @testing-library/jest-dom \
  @vitest/coverage-v8 \
  @playwright/test \
  jsdom
```

## 4. Подключить Tailwind CSS v4 (официальная инструкция)

```bash
npm install tailwindcss @tailwindcss/vite
```

В `vite.config.ts`:

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
});
```

В `src/index.css` (или другом главном CSS-файле) единственная строка:

```css
@import "tailwindcss";
```

Удали стандартные `src/App.css` или почисти от дефолтных правил Vite.

## 5. Настроить i18next с русским по умолчанию (CLAUDE.md §4)

Создай `src/i18n/index.ts`:

```ts
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import ru from "./locales/ru.json";
import en from "./locales/en.json";

await i18n.use(initReactI18next).init({
  resources: { ru: { translation: ru }, en: { translation: en } },
  lng: "ru",
  fallbackLng: "ru",
  interpolation: { escapeValue: false },
});

export default i18n;
```

Импортируй `./i18n` в `src/main.tsx` до рендера приложения.

## 6. Добавить npm-скрипты (отредактируй `package.json`)

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "lint": "eslint .",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:e2e": "playwright test"
  }
}
```

## 7. Проверка

```bash
npm run dev        # http://localhost:5173
npm run lint
npm run test
```

После выполнения всех шагов закоммить `package.json`, `package-lock.json`,
`vite.config.ts`, `tsconfig*.json`, `src/`, `index.html`, `public/`, конфиги
ESLint и Tailwind (если появятся).
