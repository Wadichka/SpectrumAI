import { useTranslation } from "react-i18next";

import type { BatchResponse } from "@/api/batch";

interface BatchSummaryProps {
  response: BatchResponse;
}

export default function BatchSummary({ response }: BatchSummaryProps) {
  const { t } = useTranslation();
  const total = response.items.length;
  const success = response.items.filter((i) => i.status === "success").length;
  const error = total - success;

  return (
    <div className="flex flex-wrap items-center gap-4 rounded-md border border-line bg-surface px-4 py-3 text-sm">
      <span className="text-ink">{t("batch.summary.total", { count: total })}</span>
      <span className="text-success">
        {t("batch.summary.success", { count: success })}
      </span>
      <span className="text-danger">
        {t("batch.summary.error", { count: error })}
      </span>
      <span className="text-muted">
        {t("batch.summary.time_ms", { ms: response.total_processing_time_ms })}
      </span>
    </div>
  );
}
