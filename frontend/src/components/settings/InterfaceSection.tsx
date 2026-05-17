import { useTranslation } from "react-i18next";

import Card from "@/components/ui/Card";
import { useSettingsStore, type Language } from "@/stores/useSettingsStore";

export default function InterfaceSection() {
  const { t, i18n } = useTranslation();
  const language = useSettingsStore((s) => s.language);
  const setLanguage = useSettingsStore((s) => s.setLanguage);

  const handleChange = (value: Language) => {
    setLanguage(value);
    void i18n.changeLanguage(value);
  };

  return (
    <Card title={t("settings.interface.title")}>
      <div className="flex flex-col gap-5">
        <fieldset className="flex flex-col gap-2 text-sm">
          <legend className="text-ink">{t("settings.interface.language")}</legend>
          <label className="flex items-center gap-2 text-ink">
            <input
              type="radio"
              name="language"
              value="ru"
              checked={language === "ru"}
              onChange={() => handleChange("ru")}
              className="accent-primary"
            />
            {t("settings.interface.language_ru")}
          </label>
          <label className="flex items-center gap-2 text-ink">
            <input
              type="radio"
              name="language"
              value="en"
              checked={language === "en"}
              onChange={() => handleChange("en")}
              className="accent-primary"
            />
            {t("settings.interface.language_en")}
          </label>
        </fieldset>
        <div className="flex flex-col gap-1 text-sm">
          <span className="text-ink">{t("settings.interface.theme")}</span>
          <span className="text-muted">{t("settings.interface.theme_stub")}</span>
        </div>
      </div>
    </Card>
  );
}
