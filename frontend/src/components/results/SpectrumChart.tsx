import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import Plot from "react-plotly.js";

import Card from "@/components/ui/Card";
import type { components } from "@/api/types.gen";

type GradCamPayload = components["schemas"]["GradCamPayload"];

interface SpectrumChartProps {
  spectrum: number[] | null;
  spectrumLength: number;
  rawSpectrum?: number[] | null;
  rawWavenumbers?: number[] | null;
  gradcam: GradCamPayload | null;
  selectedGroupCode: string | null;
}

type ViewMode = "raw" | "processed";
type IntensityMode = "absorbance" | "transmittance";

function linspace(min: number, max: number, count: number): number[] {
  if (count <= 1) return [min];
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + step * i);
}

export default function SpectrumChart({
  spectrum,
  spectrumLength,
  rawSpectrum,
  rawWavenumbers,
  gradcam,
  selectedGroupCode,
}: SpectrumChartProps) {
  const { t } = useTranslation();
  const hasRaw =
    Array.isArray(rawSpectrum) &&
    Array.isArray(rawWavenumbers) &&
    rawSpectrum.length === rawWavenumbers.length &&
    rawSpectrum.length > 1;
  // Сырой спектр понятнее химику (выглядит как на NIST WebBook), поэтому
  // он по умолчанию. Если raw недоступен (старая запись истории) —
  // автоматически переключаемся на processed.
  const [view, setView] = useState<ViewMode>(hasRaw ? "raw" : "processed");
  const [intensityMode, setIntensityMode] = useState<IntensityMode>("absorbance");

  const effectiveView: ViewMode = hasRaw ? view : "processed";

  const wavenumbers = useMemo(() => {
    if (effectiveView === "raw" && hasRaw) {
      return rawWavenumbers as number[];
    }
    return linspace(400, 4000, spectrum?.length ?? spectrumLength);
  }, [effectiveView, hasRaw, rawWavenumbers, spectrum, spectrumLength]);

  const baseIntensities = useMemo(() => {
    if (effectiveView === "raw" && hasRaw) return rawSpectrum as number[];
    return spectrum;
  }, [effectiveView, hasRaw, rawSpectrum, spectrum]);

  const display = useMemo(() => {
    if (!baseIntensities) return null;
    if (intensityMode === "absorbance") return baseIntensities;
    return baseIntensities.map((v) => Math.pow(10, -v));
  }, [baseIntensities, intensityMode]);

  const showGradcam =
    effectiveView === "processed" &&
    gradcam !== null &&
    selectedGroupCode === gradcam.group_code;

  return (
    <Card padding="sm">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-ink">{t("results.spectrum.title")}</h2>
        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex overflow-hidden rounded-md border border-line text-xs">
            <button
              type="button"
              onClick={() => setView("raw")}
              disabled={!hasRaw}
              className={`px-3 py-1.5 ${
                effectiveView === "raw"
                  ? "bg-primary text-white"
                  : "bg-surface text-muted hover:bg-background disabled:opacity-50"
              }`}
            >
              {t("results.spectrum.raw")}
            </button>
            <button
              type="button"
              onClick={() => setView("processed")}
              className={`px-3 py-1.5 ${
                effectiveView === "processed"
                  ? "bg-primary text-white"
                  : "bg-surface text-muted hover:bg-background"
              }`}
            >
              {t("results.spectrum.processed")}
            </button>
          </div>
          <div className="inline-flex overflow-hidden rounded-md border border-line text-xs">
            <button
              type="button"
              onClick={() => setIntensityMode("absorbance")}
              className={`px-3 py-1.5 ${
                intensityMode === "absorbance"
                  ? "bg-primary text-white"
                  : "bg-surface text-muted hover:bg-background"
              }`}
            >
              {t("results.spectrum.absorbance")}
            </button>
            <button
              type="button"
              onClick={() => setIntensityMode("transmittance")}
              className={`px-3 py-1.5 ${
                intensityMode === "transmittance"
                  ? "bg-primary text-white"
                  : "bg-surface text-muted hover:bg-background"
              }`}
            >
              {t("results.spectrum.transmittance")}
            </button>
          </div>
        </div>
      </div>
      {display ? (
        <Plot
          data={[
            {
              type: "scatter",
              mode: "lines",
              x: wavenumbers,
              y: display,
              line: { color: "#2563EB", width: 1.5 },
              name: t(`results.spectrum.${intensityMode}`),
              hovertemplate: "%{x:.1f} см⁻¹<br>%{y:.3f}<extra></extra>",
            },
            ...(showGradcam
              ? [
                  {
                    type: "heatmap" as const,
                    x: linspace(400, 4000, gradcam.values.length),
                    y: [0],
                    z: [gradcam.values],
                    showscale: false,
                    colorscale: "YlOrRd" as const,
                    opacity: 0.45,
                    yaxis: "y2",
                    hovertemplate: "Grad-CAM %{z:.2f}<extra></extra>",
                  },
                ]
              : []),
          ]}
          layout={{
            autosize: true,
            height: 320,
            margin: { l: 50, r: 30, t: 10, b: 40 },
            xaxis: {
              title: { text: "Волновое число, см⁻¹" },
              autorange: "reversed",
            },
            yaxis: {
              title: {
                text:
                  intensityMode === "absorbance"
                    ? t("results.spectrum.absorbance")
                    : t("results.spectrum.transmittance"),
              },
            },
            yaxis2: {
              overlaying: "y",
              showticklabels: false,
              showgrid: false,
              zeroline: false,
              range: [-0.5, 0.5],
            },
            showlegend: false,
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
          }}
          config={{ displaylogo: false, responsive: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      ) : (
        <p className="py-12 text-center text-sm text-muted">
          {t("results.spectrum.unavailable")}
        </p>
      )}
      {showGradcam ? (
        <p className="mt-2 text-xs text-muted">
          {t("results.gradcam.legend", { group: gradcam.group_name })}
        </p>
      ) : null}
    </Card>
  );
}
