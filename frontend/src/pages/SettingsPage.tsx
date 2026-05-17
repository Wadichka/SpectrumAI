import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { fetchSettings, updateSettings } from "@/api/settings";
import IdentificationSection from "@/components/settings/IdentificationSection";
import InterfaceSection from "@/components/settings/InterfaceSection";
import PreprocessingSection from "@/components/settings/PreprocessingSection";
import SettingsSidebar, {
  type SettingsSection,
} from "@/components/settings/SettingsSidebar";
import Alert from "@/components/ui/Alert";
import Button from "@/components/ui/Button";
import { useSettingsStore } from "@/stores/useSettingsStore";

type FeedbackVariant = "success" | "error";
type Feedback = { variant: FeedbackVariant; message: string };

export default function SettingsPage() {
  const { t, i18n } = useTranslation();
  const applyServerSettings = useSettingsStore((s) => s.applyServerSettings);
  const toPayload = useSettingsStore((s) => s.toPayload);
  const resetToDefaults = useSettingsStore((s) => s.resetToDefaults);

  const [section, setSection] = useState<SettingsSection>("identification");
  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState<Feedback | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchSettings()
      .then((payload) => {
        if (cancelled) return;
        applyServerSettings(payload);
        void i18n.changeLanguage(payload.language);
      })
      .catch(() => {
        if (cancelled) return;
        setFeedback({ variant: "error", message: t("settings.load_error") });
      });
    return () => {
      cancelled = true;
    };
  }, [applyServerSettings, i18n, t]);

  const handleSave = async () => {
    setSaving(true);
    setFeedback(null);
    try {
      const echoed = await updateSettings(toPayload());
      applyServerSettings(echoed);
      setFeedback({ variant: "success", message: t("settings.saved") });
    } catch {
      setFeedback({ variant: "error", message: t("settings.save_error") });
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    resetToDefaults();
    void i18n.changeLanguage("ru");
    setFeedback(null);
  };

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-bold text-ink">{t("settings.title")}</h1>

      {feedback ? (
        <Alert variant={feedback.variant} closable>
          {feedback.message}
        </Alert>
      ) : null}

      <div className="flex flex-col gap-6 md:flex-row">
        <SettingsSidebar active={section} onChange={setSection} />
        <div className="flex grow flex-col gap-4">
          {section === "identification" ? <IdentificationSection /> : null}
          {section === "interface" ? <InterfaceSection /> : null}
          {section === "preprocessing" ? <PreprocessingSection /> : null}

          <div className="flex flex-wrap gap-3">
            <Button onClick={handleSave} loading={saving}>
              {t("settings.save")}
            </Button>
            <Button variant="tertiary" onClick={handleReset} disabled={saving}>
              {t("settings.reset")}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
