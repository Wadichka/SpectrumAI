import { FileText, X } from "lucide-react";
import { useTranslation } from "react-i18next";

import Card from "@/components/ui/Card";
import { formatBytes } from "@/lib/format";

interface FilePreviewProps {
  file: File;
  onRemove: () => void;
}

export default function FilePreview({ file, onRemove }: FilePreviewProps) {
  const { t } = useTranslation();
  return (
    <Card padding="sm">
      <div className="flex items-center justify-between gap-4">
        <div className="flex min-w-0 items-center gap-3">
          <FileText className="size-6 shrink-0 text-primary" aria-hidden="true" />
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-ink">{file.name}</p>
            <p className="text-xs text-muted">
              {t("identify.preview.size", { size: formatBytes(file.size) })}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onRemove}
          aria-label={t("identify.preview.remove")}
          className="rounded p-1 text-muted hover:bg-background hover:text-ink"
        >
          <X className="size-4" aria-hidden="true" />
        </button>
      </div>
    </Card>
  );
}
