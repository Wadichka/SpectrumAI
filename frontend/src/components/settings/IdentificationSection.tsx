import { useTranslation } from "react-i18next";

import Card from "@/components/ui/Card";
import { useSettingsStore } from "@/stores/useSettingsStore";

export default function IdentificationSection() {
  const { t } = useTranslation();
  const topK = useSettingsStore((s) => s.topK);
  const threshold = useSettingsStore((s) => s.threshold);
  const includeGradcam = useSettingsStore((s) => s.includeGradcam);
  const setTopK = useSettingsStore((s) => s.setTopK);
  const setThreshold = useSettingsStore((s) => s.setThreshold);
  const setIncludeGradcam = useSettingsStore((s) => s.setIncludeGradcam);

  return (
    <Card title={t("settings.identification.title")}>
      <div className="flex flex-col gap-5">
        <label className="flex flex-col gap-2 text-sm">
          <span className="text-ink">
            {t("settings.identification.top_k")}: <strong>{topK}</strong>
          </span>
          <input
            type="range"
            min={1}
            max={50}
            step={1}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="accent-primary"
            aria-label={t("settings.identification.top_k")}
          />
        </label>
        <label className="flex flex-col gap-2 text-sm">
          <span className="text-ink">
            {t("settings.identification.threshold")}: <strong>{threshold.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="accent-primary"
            aria-label={t("settings.identification.threshold")}
          />
        </label>
        <label className="flex items-center gap-2 text-sm text-ink">
          <input
            type="checkbox"
            checked={includeGradcam}
            onChange={(e) => setIncludeGradcam(e.target.checked)}
            className="size-4 accent-primary"
          />
          {t("settings.identification.include_gradcam")}
        </label>
      </div>
    </Card>
  );
}
