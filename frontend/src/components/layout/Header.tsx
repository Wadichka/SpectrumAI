import { Globe } from "lucide-react";
import { useTranslation } from "react-i18next";

import { useSettingsStore } from "@/stores/useSettingsStore";

export default function Header() {
  const { t, i18n } = useTranslation();
  const language = useSettingsStore((s) => s.language);
  const setLanguage = useSettingsStore((s) => s.setLanguage);

  const toggleLanguage = () => {
    const next = language === "ru" ? "en" : "ru";
    setLanguage(next);
    void i18n.changeLanguage(next);
  };

  return (
    <header className="flex h-16 items-center justify-between border-b border-line bg-surface px-6">
      <div>
        <p className="text-base font-semibold text-ink">{t("app.title")}</p>
        <p className="text-xs text-muted">{t("app.tagline")}</p>
      </div>
      <button
        type="button"
        onClick={toggleLanguage}
        aria-label={t("header.language")}
        className="flex items-center gap-2 rounded-md border border-line bg-surface px-3 py-1.5 text-sm font-medium text-ink transition-colors hover:bg-background"
      >
        <Globe className="size-4" aria-hidden="true" />
        <span>{t("header.language_short")}</span>
      </button>
    </header>
  );
}
