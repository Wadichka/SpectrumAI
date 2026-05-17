import axios, { AxiosError } from "axios";
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import { postIdentifyBatch } from "@/api/batch";
import BatchDropZone from "@/components/batch/BatchDropZone";
import BatchFilesTable from "@/components/batch/BatchFilesTable";
import BatchResultsTable from "@/components/batch/BatchResultsTable";
import BatchSummary from "@/components/batch/BatchSummary";
import Alert from "@/components/ui/Alert";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import Spinner from "@/components/ui/Spinner";
import { serializeBatchToCsv } from "@/lib/batchCsv";
import { useBatchStore } from "@/stores/useBatchStore";
import { useSettingsStore } from "@/stores/useSettingsStore";

const ACCEPT_EXTENSIONS = [".jdx", ".dx", ".csv", ".txt"];
const MAX_FILES = 20;
const MAX_TOTAL_BYTES = 50 * 1024 * 1024;

function extractErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ detail?: { message?: string } | string }>;
    const detail = axiosError.response?.data?.detail;
    if (typeof detail === "object" && detail && "message" in detail && detail.message) {
      return String(detail.message);
    }
    if (typeof detail === "string") return detail;
  }
  return fallback;
}

export default function BatchPage() {
  const { t } = useTranslation();
  const topK = useSettingsStore((s) => s.topK);
  const includeGradcam = useSettingsStore((s) => s.includeGradcam);
  const files = useBatchStore((s) => s.files);
  const status = useBatchStore((s) => s.status);
  const response = useBatchStore((s) => s.response);
  const errorMessage = useBatchStore((s) => s.errorMessage);
  const addFiles = useBatchStore((s) => s.addFiles);
  const removeFile = useBatchStore((s) => s.removeFile);
  const clearFiles = useBatchStore((s) => s.clearFiles);
  const setProcessing = useBatchStore((s) => s.setProcessing);
  const setResponse = useBatchStore((s) => s.setResponse);
  const setError = useBatchStore((s) => s.setError);
  const reset = useBatchStore((s) => s.reset);

  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  const totalBytes = files.reduce((sum, f) => sum + f.size, 0);
  const tooMany = files.length > MAX_FILES;
  const tooLarge = totalBytes > MAX_TOTAL_BYTES;
  const canRun = files.length > 0 && !tooMany && !tooLarge && status !== "processing";

  const handleAddFiles = (incoming: File[]) => {
    const invalid = incoming.find(
      (f) => !ACCEPT_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext)),
    );
    if (invalid) {
      setError(t("batch.error.invalid_extension", { name: invalid.name }));
      return;
    }
    addFiles(incoming);
  };

  const handleRun = async () => {
    if (files.length === 0) {
      setError(t("batch.error.none_selected"));
      return;
    }
    if (tooMany) {
      setError(t("batch.error.too_many_files"));
      return;
    }
    if (tooLarge) {
      setError(t("batch.error.too_large_total"));
      return;
    }
    controllerRef.current = new AbortController();
    setProcessing();
    try {
      const result = await postIdentifyBatch(
        files,
        { topK, includeGradcam },
        controllerRef.current.signal,
      );
      setResponse(result);
    } catch (rawError) {
      if (axios.isCancel(rawError) || (rawError as Error)?.name === "CanceledError") {
        reset();
        return;
      }
      const message = extractErrorMessage(
        rawError,
        axios.isAxiosError(rawError) && !rawError.response
          ? t("batch.error.network")
          : t("batch.error.server"),
      );
      setError(message);
    }
  };

  const handleCancel = () => {
    controllerRef.current?.abort();
  };

  const handleExportCsv = () => {
    if (!response) return;
    const csv = serializeBatchToCsv(response.items);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
    link.download = `batch-${stamp}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const isProcessing = status === "processing";
  const isDone = status === "done" && response !== null;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold text-ink">{t("batch.title")}</h1>
        <p className="text-sm text-muted">{t("batch.subtitle")}</p>
      </div>

      {errorMessage ? (
        <Alert variant="error" closable>
          {errorMessage}
        </Alert>
      ) : null}

      {isProcessing ? (
        <Card>
          <div className="flex flex-col items-center gap-3 py-8">
            <Spinner size="lg" />
            <p className="text-base font-medium text-ink">
              {t("batch.processing.title")}
            </p>
            <p className="text-sm text-muted">
              {t("batch.processing.subtitle", { count: files.length })}
            </p>
            <Button variant="tertiary" onClick={handleCancel}>
              {t("batch.button.cancel")}
            </Button>
          </div>
        </Card>
      ) : null}

      {isDone && response ? (
        <div className="flex flex-col gap-4">
          <BatchSummary response={response} />
          <Card padding="sm">
            <BatchResultsTable items={response.items} />
          </Card>
          <div className="flex flex-wrap gap-3">
            <Button onClick={handleExportCsv}>{t("batch.button.export_csv")}</Button>
            <Button variant="tertiary" onClick={reset}>
              {t("batch.button.new_batch")}
            </Button>
          </div>
        </div>
      ) : null}

      {!isProcessing && !isDone ? (
        <div className="flex flex-col gap-4">
          <BatchDropZone onFiles={handleAddFiles} accept={ACCEPT_EXTENSIONS} />
          <Card>
            <BatchFilesTable
              files={files}
              onRemove={removeFile}
              onClearAll={clearFiles}
            />
            {tooMany ? (
              <p className="mt-2 text-sm text-danger">
                {t("batch.error.too_many_files")}
              </p>
            ) : null}
            {tooLarge ? (
              <p className="mt-2 text-sm text-danger">
                {t("batch.error.too_large_total")}
              </p>
            ) : null}
          </Card>
          <div className="flex flex-wrap gap-3">
            <Button onClick={handleRun} disabled={!canRun}>
              {t("batch.button.run")}
            </Button>
            <Button
              variant="tertiary"
              onClick={clearFiles}
              disabled={files.length === 0}
            >
              {t("batch.button.clear")}
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
