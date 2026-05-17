import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import Plot from "react-plotly.js";

import Card from "@/components/ui/Card";
import type { components } from "@/api/types.gen";

type GradCamPayload = components["schemas"]["GradCamPayload"];

interface SpectrumChartProps {
  spectrum: number[] | null;
  spectrumLength: number;
  gradcam: GradCamPayload | null;
  selectedGroupCode: string | null;
}

function linspace(min: number, max: number, count: number): number[] {
  if (count <= 1) return [min];
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + step * i);
}

export default function SpectrumChart({
  spectrum,
  spectrumLength,
  gradcam,
  selectedGroupCode,
}: SpectrumChartProps) {
  const { t } = useTranslation();
  const [mode, setMode] = useState<"absorbance" | "transmittance">("absorbance");

  const wavenumbers = useMemo(
    () => linspace(400, 4000, spectrum?.length ?? spectrumLength),
    [spectrum, spectrumLength],
  );

  const display = useMemo(() => {
    if (!spectrum) return null;
    if (mode === "absorbance") return spectrum;
    return spectrum.map((v) => Math.pow(10, -v));
  }, [spectrum, mode]);

  const showGradcam = gradcam !== null && selectedGroupCode === gradcam.group_code;

  return (
    <Card padding="sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-ink">{t("results.spectrum.title")}</h2>
        <div className="inline-flex overflow-hidden rounded-md border border-line text-xs">
          <button
            type="button"
            onClick={() => setMode("absorbance")}
            className={`px-3 py-1.5 ${
              mode === "absorbance"
                ? "bg-primary text-white"
                : "bg-surface text-muted hover:bg-background"
            }`}
          >
            {t("results.spectrum.absorbance")}
          </button>
          <button
            type="button"
            onClick={() => setMode("transmittance")}
            className={`px-3 py-1.5 ${
              mode === "transmittance"
                ? "bg-primary text-white"
                : "bg-surface text-muted hover:bg-background"
            }`}
          >
            {t("results.spectrum.transmittance")}
          </button>
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
              name: t(`results.spectrum.${mode}`),
              hovertemplate: "%{x:.0f} см⁻¹<br>%{y:.3f}<extra></extra>",
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
                  mode === "absorbance"
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
