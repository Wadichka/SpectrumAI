import { X } from "lucide-react";
import { useTranslation } from "react-i18next";

import Button from "@/components/ui/Button";
import { formatBytes } from "@/lib/format";

interface BatchFilesTableProps {
  files: File[];
  onRemove: (index: number) => void;
  onClearAll: () => void;
  disabled?: boolean;
}

export default function BatchFilesTable({
  files,
  onRemove,
  onClearAll,
  disabled,
}: BatchFilesTableProps) {
  const { t } = useTranslation();

  if (files.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-muted">{t("batch.files.empty")}</p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-ink">
          {t("batch.files.title", { count: files.length })}
        </span>
        <Button
          size="sm"
          variant="tertiary"
          onClick={onClearAll}
          disabled={disabled}
        >
          {t("batch.files.clear_all")}
        </Button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase text-muted">
            <tr>
              <th className="px-3 py-2">{t("batch.files.columns.name")}</th>
              <th className="px-3 py-2">{t("batch.files.columns.size")}</th>
              <th className="px-3 py-2 text-right">
                {t("batch.files.columns.actions")}
              </th>
            </tr>
          </thead>
          <tbody>
            {files.map((file, index) => (
              <tr
                key={`${file.name}-${index}`}
                className="border-t border-line hover:bg-background"
              >
                <td className="px-3 py-2 align-middle text-ink">{file.name}</td>
                <td className="px-3 py-2 align-middle text-muted">
                  {formatBytes(file.size)}
                </td>
                <td className="px-3 py-2 text-right align-middle">
                  <button
                    type="button"
                    aria-label={t("batch.files.remove")}
                    onClick={() => onRemove(index)}
                    disabled={disabled}
                    className="rounded p-1 text-muted transition-colors hover:bg-line hover:text-danger disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <X className="size-4" aria-hidden="true" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
