import type { Config } from "tailwindcss";

// Дизайн-токены: глава 7 §7.3.1 (цветовая палитра), §7.3.2 (типографика),
// §7.3.4 (сетка 8px, max-width 1280).

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#2563EB",
          hover: "#1D4ED8",
          muted: "#DBEAFE",
        },
        secondary: {
          DEFAULT: "#10B981",
          hover: "#059669",
        },
        surface: "#FFFFFF",
        background: "#F9FAFB",
        line: "#E5E7EB",
        muted: "#6B7280",
        ink: "#111827",
        danger: "#EF4444",
        warning: "#F59E0B",
        success: "#10B981",
        info: "#3B82F6",
      },
      fontFamily: {
        sans: ['"Inter"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      maxWidth: {
        container: "1280px",
      },
      spacing: {
        gutter: "1.5rem",
      },
    },
  },
  plugins: [],
} satisfies Config;
