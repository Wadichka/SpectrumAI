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

interface DropZoneProps {
  onFile: (file: File) => void;
  accept: string[];
  ariaLabel?: string;
}

export default function DropZone({ onFile, accept, ariaLabel }: DropZoneProps) {
  const { t } = useTranslation();
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const openDialog = useCallback(() => inputRef.current?.click(), []);

  const handleKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openDialog();
    }
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) onFile(file);
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={ariaLabel ?? t("identify.dropzone.idle")}
      onClick={openDialog}
      onKeyDown={handleKey}
      onDragOver={handleDragOver}
      onDragEnter={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "flex min-h-[240px] cursor-pointer flex-col items-center justify-center gap-3 rounded-lg",
        "border-2 border-dashed transition-colors",
        dragging
          ? "border-primary bg-primary-muted"
          : "border-primary/40 bg-surface hover:border-primary",
      )}
    >
      <Upload className="size-10 text-primary" aria-hidden="true" />
      <p className="text-base font-medium text-ink">
        {dragging ? t("identify.dropzone.dragging") : t("identify.dropzone.idle")}
      </p>
      <p className="text-xs text-muted">{t("identify.dropzone.formats")}</p>
      <input
        ref={inputRef}
        type="file"
        accept={accept.join(",")}
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
          event.target.value = "";
        }}
      />
    </div>
  );
}
