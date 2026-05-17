import { Check, Circle } from "lucide-react";
import { useTranslation } from "react-i18next";

import Spinner from "@/components/ui/Spinner";
import { cn } from "@/lib/cn";

const STEP_KEYS = [
  "identify.steps.parsing",
  "identify.steps.preprocessing",
  "identify.steps.analyzing",
  "identify.steps.searching",
  "identify.steps.ready",
] as const;

export interface ProcessingStepperProps {
  currentStep: number; // 0..4
  filename?: string | null;
}

export default function ProcessingStepper({ currentStep, filename }: ProcessingStepperProps) {
  const { t } = useTranslation();
  return (
    <section
      className="flex flex-col gap-6 rounded-lg border border-line bg-surface p-6"
      aria-live="polite"
    >
      <header className="flex flex-col gap-1">
        <h2 className="text-lg font-semibold text-ink">{t("identify.processing.title")}</h2>
        {filename ? (
          <p className="font-mono text-sm text-muted">
            {t("identify.processing.subtitle", { name: filename })}
          </p>
        ) : null}
      </header>
      <ol className="flex flex-col gap-3" role="list">
        {STEP_KEYS.map((key, index) => {
          const status =
            index < currentStep ? "done" : index === currentStep ? "active" : "pending";
          return (
            <li
              key={key}
              className={cn(
                "flex items-center gap-3 text-sm",
                status === "done" && "text-success",
                status === "active" && "font-medium text-ink",
                status === "pending" && "text-muted",
              )}
            >
              <span className="flex size-6 items-center justify-center">
                {status === "done" ? (
                  <Check className="size-5 text-success" aria-hidden="true" />
                ) : status === "active" ? (
                  <Spinner size="sm" className="text-primary" aria-hidden="true" />
                ) : (
                  <Circle className="size-4 text-muted" aria-hidden="true" />
                )}
              </span>
              <span>{t(key)}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
