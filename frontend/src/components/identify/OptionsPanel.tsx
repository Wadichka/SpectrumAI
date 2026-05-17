import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

interface OptionsPanelProps {
  topK: number;
  includeGradcam: boolean;
  onTopKChange: (value: number) => void;
  onIncludeGradcamChange: (value: boolean) => void;
}

export default function OptionsPanel({
  topK,
  includeGradcam,
  onTopKChange,
  onIncludeGradcamChange,
}: OptionsPanelProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);

  return (
    <section className="rounded-lg border border-line bg-surface">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-center justify-between rounded-lg px-4 py-3 text-left text-sm font-medium text-ink hover:bg-background"
      >
        <span className="flex items-center gap-2">
          {open ? (
            <ChevronDown className="size-4 text-muted" aria-hidden="true" />
          ) : (
            <ChevronRight className="size-4 text-muted" aria-hidden="true" />
          )}
          {t("identify.options.title")}
        </span>
      </button>
      {open ? (
        <div className="flex flex-col gap-4 border-t border-line px-4 py-4">
          <label className="flex flex-col gap-2 text-sm">
            <span className="text-ink">
              {t("identify.options.top_k")}: <strong>{topK}</strong>
            </span>
            <input
              type="range"
              min={1}
              max={50}
              step={1}
              value={topK}
              onChange={(event) => onTopKChange(Number(event.target.value))}
              className="accent-primary"
              aria-label={t("identify.options.top_k")}
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-ink">
            <input
              type="checkbox"
              checked={includeGradcam}
              onChange={(event) => onIncludeGradcamChange(event.target.checked)}
              className="size-4 accent-primary"
            />
            {t("identify.options.include_gradcam")}
          </label>
        </div>
      ) : null}
    </section>
  );
}
