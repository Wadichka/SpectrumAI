import { useTranslation } from "react-i18next";

import Button from "@/components/ui/Button";

export interface HistoryFiltersValue {
  date_from: string;
  date_to: string;
  status: string;
}

interface HistoryFiltersProps {
  value: HistoryFiltersValue;
  onChange: (value: HistoryFiltersValue) => void;
  onReset: () => void;
}

export default function HistoryFilters({ value, onChange, onReset }: HistoryFiltersProps) {
  const { t } = useTranslation();

  const handleField = (key: keyof HistoryFiltersValue, raw: string) => {
    onChange({ ...value, [key]: raw });
  };

  return (
    <div className="flex flex-wrap items-end gap-3">
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-muted">{t("history.filters.date_from")}</span>
        <input
          type="date"
          value={value.date_from}
          onChange={(e) => handleField("date_from", e.target.value)}
          className="rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink"
        />
      </label>
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-muted">{t("history.filters.date_to")}</span>
        <input
          type="date"
          value={value.date_to}
          onChange={(e) => handleField("date_to", e.target.value)}
          className="rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink"
        />
      </label>
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-muted">{t("history.filters.status")}</span>
        <select
          value={value.status}
          onChange={(e) => handleField("status", e.target.value)}
          className="rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink"
        >
          <option value="all">{t("history.filters.status_all")}</option>
          <option value="success">{t("history.filters.status_success")}</option>
          <option value="error">{t("history.filters.status_error")}</option>
          <option value="pending">{t("history.filters.status_pending")}</option>
        </select>
      </label>
      <Button variant="tertiary" size="sm" onClick={onReset}>
        {t("history.filters.reset")}
      </Button>
    </div>
  );
}
