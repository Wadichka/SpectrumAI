import { useTranslation } from "react-i18next";

import Card from "@/components/ui/Card";
import {
  useSettingsStore,
  type BaselineMethod,
  type NormalizeMethod,
} from "@/stores/useSettingsStore";

export default function PreprocessingSection() {
  const { t } = useTranslation();
  const baselineMethod = useSettingsStore((s) => s.baselineMethod);
  const normalizeMethod = useSettingsStore((s) => s.normalizeMethod);
  const savgolWindow = useSettingsStore((s) => s.savgolWindow);
  const savgolPolyorder = useSettingsStore((s) => s.savgolPolyorder);
  const setBaselineMethod = useSettingsStore((s) => s.setBaselineMethod);
  const setNormalizeMethod = useSettingsStore((s) => s.setNormalizeMethod);
  const setSavgolWindow = useSettingsStore((s) => s.setSavgolWindow);
  const setSavgolPolyorder = useSettingsStore((s) => s.setSavgolPolyorder);

  return (
    <Card title={t("settings.preprocessing.title")}>
      <div className="flex flex-col gap-5">
        <fieldset className="flex flex-col gap-2 text-sm">
          <legend className="text-ink">{t("settings.preprocessing.baseline")}</legend>
          {(["asls", "none"] as BaselineMethod[]).map((value) => (
            <label key={value} className="flex items-center gap-2 text-ink">
              <input
                type="radio"
                name="baseline"
                value={value}
                checked={baselineMethod === value}
                onChange={() => setBaselineMethod(value)}
                className="accent-primary"
              />
              {t(`settings.preprocessing.baseline_${value}`)}
            </label>
          ))}
        </fieldset>

        <fieldset className="flex flex-col gap-2 text-sm">
          <legend className="text-ink">{t("settings.preprocessing.normalize")}</legend>
          {(["snv", "minmax"] as NormalizeMethod[]).map((value) => (
            <label key={value} className="flex items-center gap-2 text-ink">
              <input
                type="radio"
                name="normalize"
                value={value}
                checked={normalizeMethod === value}
                onChange={() => setNormalizeMethod(value)}
                className="accent-primary"
              />
              {t(`settings.preprocessing.normalize_${value}`)}
            </label>
          ))}
        </fieldset>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-ink">{t("settings.preprocessing.savgol_window")}</span>
          <input
            type="number"
            min={3}
            max={51}
            step={2}
            value={savgolWindow}
            onChange={(e) => setSavgolWindow(Number(e.target.value))}
            className="w-32 rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-ink">{t("settings.preprocessing.savgol_polyorder")}</span>
          <input
            type="number"
            min={0}
            max={5}
            step={1}
            value={savgolPolyorder}
            onChange={(e) => setSavgolPolyorder(Number(e.target.value))}
            className="w-32 rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink"
          />
        </label>
      </div>
    </Card>
  );
}
