import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { structureSvgUrlBySmiles, type CompoundSummary } from "@/api/compounds";
import Button from "@/components/ui/Button";

interface CompoundsTableProps {
  rows: CompoundSummary[];
}

export default function CompoundsTable({ rows }: CompoundsTableProps) {
  const { t } = useTranslation();
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs uppercase text-muted">
          <tr>
            <th className="px-3 py-2">{t("database.columns.structure")}</th>
            <th className="px-3 py-2">{t("database.columns.name")}</th>
            <th className="px-3 py-2">{t("database.columns.formula")}</th>
            <th className="px-3 py-2">{t("database.columns.smiles")}</th>
            <th className="px-3 py-2">{t("database.columns.cas")}</th>
            <th className="px-3 py-2 text-right">{t("database.columns.actions")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className="border-t border-line hover:bg-background">
              <td className="px-3 py-2 align-middle">
                <img
                  src={structureSvgUrlBySmiles(row.smiles, { width: 96, height: 72 })}
                  alt={row.name ?? row.smiles}
                  className="h-16 w-24 rounded border border-line bg-surface object-contain"
                  loading="lazy"
                />
              </td>
              <td className="px-3 py-2 align-middle text-ink">
                {row.name ?? t("database.no_name")}
              </td>
              <td className="px-3 py-2 align-middle text-ink">{row.formula ?? "—"}</td>
              <td className="px-3 py-2 align-middle font-mono text-xs text-muted">
                {row.smiles}
              </td>
              <td className="px-3 py-2 align-middle text-muted">{row.cas_number ?? "—"}</td>
              <td className="px-3 py-2 text-right align-middle">
                <Link to={`/compounds/${row.id}`}>
                  <Button size="sm" variant="tertiary">
                    {t("database.open")}
                  </Button>
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
