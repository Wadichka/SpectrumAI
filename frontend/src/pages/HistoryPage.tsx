import { History as HistoryIcon } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

import { fetchHistory, type HistoryEntry, type PaginatedHistory } from "@/api/history";
import HistoryFilters, {
  type HistoryFiltersValue,
} from "@/components/history/HistoryFilters";
import Alert from "@/components/ui/Alert";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import Spinner from "@/components/ui/Spinner";

const PAGE_SIZES = [20, 50, 100] as const;
const DEFAULT_FILTERS: HistoryFiltersValue = { date_from: "", date_to: "", status: "all" };

function formatTimestamp(iso: string, locale: string): string {
  return new Intl.DateTimeFormat(locale, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(iso));
}

function statusVariant(status: string): "success" | "warning" | "error" | "neutral" {
  if (status === "success") return "success";
  if (status === "error") return "error";
  if (status === "pending") return "warning";
  return "neutral";
}

export default function HistoryPage() {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();

  const [filters, setFilters] = useState<HistoryFiltersValue>(DEFAULT_FILTERS);
  const [page, setPage] = useState(1);
  const [size, setSize] = useState<number>(20);
  const [data, setData] = useState<PaginatedHistory | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchHistory({
        page,
        size,
        date_from: filters.date_from || undefined,
        date_to: filters.date_to || undefined,
        status: filters.status,
      });
      setData(result);
    } catch {
      setError(t("history.load_error"));
    } finally {
      setLoading(false);
    }
  }, [filters, page, size, t]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const handleResetFilters = () => {
    setFilters(DEFAULT_FILTERS);
    setPage(1);
  };

  const handleFiltersChange = (next: HistoryFiltersValue) => {
    setFilters(next);
    setPage(1);
  };

  const handleOpen = (entry: HistoryEntry) => {
    navigate(`/identify/results/${entry.request_id}`);
  };

  const rows: HistoryEntry[] = data?.data ?? [];
  const total = data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / size));

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-bold text-ink">{t("history.title")}</h1>

      <Card>
        <HistoryFilters
          value={filters}
          onChange={handleFiltersChange}
          onReset={handleResetFilters}
        />
      </Card>

      {error ? <Alert variant="error">{error}</Alert> : null}

      <Card padding="sm">
        {loading ? (
          <div className="flex items-center justify-center py-10">
            <Spinner />
          </div>
        ) : rows.length === 0 ? (
          <div className="flex flex-col items-center gap-3 py-10 text-center">
            <HistoryIcon className="size-10 text-muted" aria-hidden="true" />
            <p className="text-base font-semibold text-ink">{t("history.empty.title")}</p>
            <p className="text-sm text-muted">{t("history.empty.description")}</p>
            <Button onClick={() => navigate("/identify")}>{t("history.empty.cta")}</Button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase text-muted">
                <tr>
                  <th className="px-3 py-2">{t("history.columns.timestamp")}</th>
                  <th className="px-3 py-2">{t("history.columns.filename")}</th>
                  <th className="px-3 py-2">{t("history.columns.top_group")}</th>
                  <th className="px-3 py-2">{t("history.columns.status")}</th>
                  <th className="px-3 py-2">{t("history.columns.processing_time")}</th>
                  <th className="px-3 py-2 text-right">{t("history.columns.actions")}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr
                    key={row.request_id}
                    className="border-t border-line hover:bg-background"
                  >
                    <td className="px-3 py-2 align-middle text-ink">
                      {formatTimestamp(row.timestamp, i18n.language)}
                    </td>
                    <td className="px-3 py-2 align-middle text-ink">
                      {row.input_filename ?? "—"}
                    </td>
                    <td className="px-3 py-2 align-middle text-ink">
                      {row.top_predicted_groups.length > 0
                        ? row.top_predicted_groups[0]
                        : t("history.no_top_group")}
                    </td>
                    <td className="px-3 py-2 align-middle">
                      <Badge variant={statusVariant(row.status)}>{row.status}</Badge>
                    </td>
                    <td className="px-3 py-2 align-middle text-muted">
                      {row.processing_time_ms != null ? `${row.processing_time_ms} ms` : "—"}
                    </td>
                    <td className="px-3 py-2 text-right align-middle">
                      <Button size="sm" variant="tertiary" onClick={() => handleOpen(row)}>
                        {t("history.open")}
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {rows.length > 0 ? (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3 text-sm">
            <label className="flex items-center gap-2">
              <span className="text-muted">{t("history.pagination.page_size")}</span>
              <select
                value={size}
                onChange={(e) => {
                  setPage(1);
                  setSize(Number(e.target.value));
                }}
                className="rounded-md border border-line bg-surface px-2 py-1 text-sm text-ink"
              >
                {PAGE_SIZES.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
            <span className="text-muted">
              {t("history.pagination.summary", { page, pages, total })}
            </span>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="tertiary"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
              >
                {t("history.pagination.prev")}
              </Button>
              <Button
                size="sm"
                variant="tertiary"
                onClick={() => setPage((p) => Math.min(pages, p + 1))}
                disabled={page >= pages}
              >
                {t("history.pagination.next")}
              </Button>
            </div>
          </div>
        ) : null}
      </Card>
    </div>
  );
}
