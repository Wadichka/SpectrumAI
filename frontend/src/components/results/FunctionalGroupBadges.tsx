import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import Badge from "@/components/ui/Badge";
import Card from "@/components/ui/Card";
import type { components } from "@/api/types.gen";
import { cn } from "@/lib/cn";

type FunctionalGroupPrediction = components["schemas"]["FunctionalGroupPrediction"];

interface FunctionalGroupBadgesProps {
  predictions: FunctionalGroupPrediction[];
  selectedCode: string | null;
  onSelect: (code: string | null) => void;
}

function variantFor(prediction: FunctionalGroupPrediction): "success" | "warning" | "info" | "neutral" {
  if (!prediction.predicted) return "neutral";
  if (prediction.probability >= 0.8) return "success";
  if (prediction.probability >= 0.5) return "warning";
  return "info";
}

export default function FunctionalGroupBadges({
  predictions,
  selectedCode,
  onSelect,
}: FunctionalGroupBadgesProps) {
  const { t } = useTranslation();
  const [showAll, setShowAll] = useState(false);

  const sorted = useMemo(
    () => [...predictions].sort((a, b) => b.probability - a.probability),
    [predictions],
  );
  const predictedCount = sorted.filter((p) => p.predicted).length;
  const visible = showAll ? sorted : sorted.slice(0, Math.max(predictedCount, 8));

  return (
    <Card padding="sm">
      <header className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-base font-semibold text-ink">{t("results.predictions.title")}</h2>
        <button
          type="button"
          onClick={() => setShowAll((value) => !value)}
          className="text-xs font-medium text-primary hover:underline"
        >
          {showAll ? t("results.predictions.show_predicted") : t("results.predictions.show_all")}
        </button>
      </header>
      <div className="flex flex-wrap gap-2">
        {visible.map((prediction) => {
          const isSelected = selectedCode === prediction.code;
          return (
            <button
              key={prediction.code}
              type="button"
              onClick={() => onSelect(isSelected ? null : prediction.code)}
              className={cn(
                "rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                isSelected ? "ring-2 ring-primary ring-offset-1" : "",
              )}
            >
              <Badge variant={variantFor(prediction)}>
                {prediction.name} ({(prediction.probability * 100).toFixed(0)}%)
              </Badge>
            </button>
          );
        })}
      </div>
    </Card>
  );
}
