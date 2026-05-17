import { Upload } from "lucide-react";
import {
  useCallback,
  useRef,
  useState,
  type DragEvent,
  type KeyboardEvent,
} from "react";
import { useTranslation } from "react-i18next";

import { cn } from "@/lib/cn";

interface BatchDropZoneProps {
  onFiles: (files: File[]) => void;
  accept: string[];
  disabled?: boolean;
}

export default function BatchDropZone({ onFiles, accept, disabled }: BatchDropZoneProps) {
  const { t } = useTranslation();
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const openDialog = useCallback(() => {
    if (disabled) return;
    inputRef.current?.click();
  }, [disabled]);

  const handleKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openDialog();
    }
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (disabled) return;
    setDragging(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (disabled) return;
    const list = Array.from(event.dataTransfer.files ?? []);
    if (list.length > 0) onFiles(list);
  };

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled || undefined}
      aria-label={t("batch.dropzone.idle")}
      onClick={openDialog}
      onKeyDown={handleKey}
      onDragOver={handleDragOver}
      onDragEnter={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "flex min-h-[200px] cursor-pointer flex-col items-center justify-center gap-3 rounded-lg",
        "border-2 border-dashed transition-colors",
        disabled
          ? "cursor-not-allowed border-line bg-background opacity-60"
          : dragging
            ? "border-primary bg-primary-muted"
            : "border-primary/40 bg-surface hover:border-primary",
      )}
    >
      <Upload className="size-10 text-primary" aria-hidden="true" />
      <p className="text-base font-medium text-ink">
        {dragging ? t("batch.dropzone.dragging") : t("batch.dropzone.idle")}
      </p>
      <p className="text-xs text-muted">{t("batch.dropzone.formats")}</p>
      <p className="text-xs text-muted">{t("batch.dropzone.limits")}</p>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={accept.join(",")}
        className="hidden"
        onChange={(event) => {
          const list = Array.from(event.target.files ?? []);
          if (list.length > 0) onFiles(list);
          event.target.value = "";
        }}
      />
    </div>
  );
}
