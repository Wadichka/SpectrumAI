import axios, { AxiosError } from "axios";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

import { postIdentify, type ApiError } from "@/api/identify";
import DropZone from "@/components/identify/DropZone";
import FilePreview from "@/components/identify/FilePreview";
import OptionsPanel from "@/components/identify/OptionsPanel";
import ProcessingStepper from "@/components/identify/ProcessingStepper";
import Alert from "@/components/ui/Alert";
import Button from "@/components/ui/Button";
import { useIdentificationStore } from "@/stores/useIdentificationStore";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { useUploadStore } from "@/stores/useUploadStore";

const ACCEPT_EXTENSIONS = [".jdx", ".dx", ".csv", ".txt"];
const MAX_BYTES = 10 * 1024 * 1024;

type ValidationCode = "INVALID_EXTENSION" | "TOO_LARGE";

function validateFile(file: File): { ok: true } | { ok: false; code: ValidationCode } {
  const lower = file.name.toLowerCase();
  if (!ACCEPT_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
    return { ok: false, code: "INVALID_EXTENSION" };
  }
  if (file.size > MAX_BYTES) {
    return { ok: false, code: "TOO_LARGE" };
  }
  return { ok: true };
}

export default function IdentificationPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const file = useUploadStore((s) => s.file);
  const setFile = useUploadStore((s) => s.setFile);
  const state = useIdentificationStore((s) => s.state);
  const setState = useIdentificationStore((s) => s.setState);
  const errorMessage = useIdentificationStore((s) => s.errorMessage);
  const setError = useIdentificationStore((s) => s.setError);
  const setLastRequestId = useIdentificationStore((s) => s.setLastRequestId);
  const setLastResponse = useIdentificationStore((s) => s.setLastResponse);
  const resetIdentification = useIdentificationStore((s) => s.reset);

  const topK = useSettingsStore((s) => s.topK);
  const includeGradcam = useSettingsStore((s) => s.includeGradcam);
  const setTopK = useSettingsStore((s) => s.setTopK);
  const setIncludeGradcam = useSettingsStore((s) => s.setIncludeGradcam);
  const [currentStep, setCurrentStep] = useState(0);
  const [showCancel, setShowCancel] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const timersRef = useRef<number[]>([]);

  useEffect(() => {
    return () => {
      timersRef.current.forEach((id) => window.clearTimeout(id));
      controllerRef.current?.abort();
    };
  }, []);

  const handleFile = (selected: File) => {
    const result = validateFile(selected);
    if (!result.ok) {
      setError(
        result.code === "INVALID_EXTENSION"
          ? t("identify.error.invalid_extension")
          : t("identify.error.too_large"),
      );
      return;
    }
    setFile(selected);
    resetIdentification();
  };

  const handleRemove = () => {
    setFile(null);
    resetIdentification();
  };

  const handleUseExample = async () => {
    try {
      const response = await fetch("/example.jdx");
      const blob = await response.blob();
      const exampleFile = new File([blob], "example.jdx", {
        type: "chemical/x-jcamp-dx",
      });
      handleFile(exampleFile);
    } catch {
      setError(t("identify.error.network"));
    }
  };

  const cleanupTimers = () => {
    timersRef.current.forEach((id) => window.clearTimeout(id));
    timersRef.current = [];
  };

  const startSimulation = () => {
    setCurrentStep(0);
    timersRef.current.push(window.setTimeout(() => setCurrentStep(1), 100));
    timersRef.current.push(window.setTimeout(() => setCurrentStep(2), 350));
    timersRef.current.push(window.setTimeout(() => setCurrentStep(3), 600));
    timersRef.current.push(window.setTimeout(() => setShowCancel(true), 1000));
  };

  const handleSubmit = async () => {
    if (!file) return;
    const validation = validateFile(file);
    if (!validation.ok) {
      setError(
        validation.code === "INVALID_EXTENSION"
          ? t("identify.error.invalid_extension")
          : t("identify.error.too_large"),
      );
      return;
    }
    cleanupTimers();
    setShowCancel(false);
    controllerRef.current = new AbortController();
    setState("preprocessing");
    startSimulation();
    try {
      const response = await postIdentify(
        file,
        { includeGradcam, topK },
        controllerRef.current.signal,
      );
      cleanupTimers();
      setCurrentStep(4);
      const requestId = response.request_id;
      setLastRequestId(requestId ?? null);
      setLastResponse(response);
      setState("results");
      window.setTimeout(() => {
        if (requestId != null) {
          navigate(`/identify/results/${requestId}`);
        } else {
          navigate("/identify/results");
        }
      }, 300);
    } catch (rawError) {
      cleanupTimers();
      if (axios.isCancel(rawError) || (rawError as Error)?.name === "CanceledError") {
        resetIdentification();
        return;
      }
      const message = extractErrorMessage(rawError, t);
      setError(message);
    }
  };

  const handleCancel = () => {
    controllerRef.current?.abort();
    cleanupTimers();
    setShowCancel(false);
    resetIdentification();
  };

  const inProgress =
    state === "preprocessing" ||
    state === "analyzing" ||
    state === "searching" ||
    state === "results";

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-bold text-ink">{t("identify.title")}</h1>

      {state === "error" && errorMessage ? (
        <Alert variant="error" closable>
          {errorMessage}
        </Alert>
      ) : null}

      {inProgress ? (
        <>
          <ProcessingStepper currentStep={currentStep} filename={file?.name ?? null} />
          {showCancel ? (
            <div>
              <Button variant="tertiary" onClick={handleCancel}>
                {t("identify.button.cancel")}
              </Button>
            </div>
          ) : null}
        </>
      ) : (
        <div className="flex flex-col gap-4">
          {file ? (
            <FilePreview file={file} onRemove={handleRemove} />
          ) : (
            <DropZone onFile={handleFile} accept={ACCEPT_EXTENSIONS} />
          )}
          <OptionsPanel
            topK={topK}
            includeGradcam={includeGradcam}
            onTopKChange={setTopK}
            onIncludeGradcamChange={setIncludeGradcam}
          />
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={handleSubmit} disabled={!file}>
              {t("identify.button.identify")}
            </Button>
            <Button variant="tertiary" onClick={handleUseExample}>
              {t("identify.button.use_example")}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function extractErrorMessage(error: unknown, t: (key: string) => string): string {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ detail?: ApiError | string }>;
    const status = axiosError.response?.status;
    const detail = axiosError.response?.data?.detail;
    if (typeof detail === "object" && detail && "message" in detail && detail.message) {
      return String(detail.message);
    }
    if (typeof detail === "string") {
      return detail;
    }
    if (status && status >= 500) return t("identify.error.server");
    if (!status) return t("identify.error.network");
    return t("identify.error.server");
  }
  return t("identify.error.network");
}
