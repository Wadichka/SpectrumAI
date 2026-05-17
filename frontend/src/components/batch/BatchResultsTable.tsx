import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import type { BatchItem } from "@/api/batch";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";

interface BatchResultsTableProps {
  items: BatchItem[];
}

export default function BatchResultsTable({ items }: BatchResultsTableProps) {
  const { t } = useTranslation();

  if (items.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-muted">{t("batch.results.empty")}</p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs uppercase text-muted">
          <tr>
            <th className="px-3 py-2">{t("batch.results.columns.filename")}</th>
            <th className="px-3 py-2">{t("batch.results.columns.status")}</th>
            <th className="px-3 py-2">{t("batch.results.columns.top_group")}</th>
            <th className="px-3 py-2">{t("batch.results.columns.score")}</th>
            <th className="px-3 py-2">{t("batch.results.columns.processing_time")}</th>
            <th className="px-3 py-2 text-right">
              {t("batch.results.columns.actions")}
            </th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, index) => {
            const top1 = item.result?.candidates?.[0] ?? null;
            const requestId = item.result?.request_id ?? null;
            return (
              <tr
                key={`${item.filename}-${index}`}
                className="border-t border-line hover:bg-background"
              >
                <td className="px-3 py-2 align-middle text-ink">{item.filename}</td>
                <td className="px-3 py-2 align-middle">
                  <Badge variant={item.status === "success" ? "success" : "error"}>
                    {t(`batch.status.${item.status}`)}
                  </Badge>
                </td>
                <td className="px-3 py-2 align-middle text-ink">
                  {top1?.name ?? top1?.smiles ?? t("batch.results.no_top_group")}
                </td>
                <td className="px-3 py-2 align-middle text-muted">
                  {top1 ? top1.score.toFixed(3) : "—"}
                </td>
                <td className="px-3 py-2 align-middle text-muted">
                  {item.result?.processing_time_ms != null
                    ? `${item.result.processing_time_ms} ms`
                    : "—"}
                </td>
                <td className="px-3 py-2 text-right align-middle">
                  {requestId != null ? (
                    <Link to={`/identify/results/${requestId}`}>
                      <Button size="sm" variant="tertiary">
                        {t("batch.results.open")}
                      </Button>
                    </Link>
                  ) : null}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
