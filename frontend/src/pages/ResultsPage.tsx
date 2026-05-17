import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import CandidateCard from "@/components/results/CandidateCard";
import FunctionalGroupBadges from "@/components/results/FunctionalGroupBadges";
import ResultsActions from "@/components/results/ResultsActions";
import SpectrumChart from "@/components/results/SpectrumChart";
import Alert from "@/components/ui/Alert";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import { useIdentificationStore } from "@/stores/useIdentificationStore";

export default function ResultsPage() {
  const { t } = useTranslation();
  const response = useIdentificationStore((s) => s.lastResponse);
  const [selectedCode, setSelectedCode] = useState<string | null>(
    response?.gradcam?.group_code ?? null,
  );

  const sortedCandidates = useMemo(
    () => (response ? [...response.candidates].sort((a, b) => a.rank - b.rank) : []),
    [response],
  );

  if (!response) {
    return (
      <div className="flex flex-col gap-4">
        <h1 className="text-2xl font-bold text-ink">{t("results.title")}</h1>
        <Alert variant="warning" title={t("results.lost_session")}>
          <Link to="/identify">
            <Button variant="primary">{t("results.lost_session.cta")}</Button>
          </Link>
        </Alert>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-ink">{t("results.title")}</h1>
          <p className="text-sm text-muted">
            {t("results.subtitle", {
              ms: response.processing_time_ms,
              mode: response.model_versions["mode"] ?? "n/a",
            })}
          </p>
        </div>
      </header>

      <SpectrumChart
        spectrum={response.spectrum ?? null}
        spectrumLength={response.spectrum_length}
        gradcam={response.gradcam ?? null}
        selectedGroupCode={selectedCode}
      />

      <FunctionalGroupBadges
        predictions={response.predictions}
        selectedCode={selectedCode}
        onSelect={setSelectedCode}
      />

      <section>
        <h2 className="mb-3 text-lg font-semibold text-ink">{t("results.candidates.title")}</h2>
        {sortedCandidates.length === 0 ? (
          <Card padding="sm">
            <p className="text-sm text-muted">{t("results.candidates.empty")}</p>
          </Card>
        ) : (
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            {sortedCandidates.map((candidate) => (
              <CandidateCard key={candidate.compound_id} candidate={candidate} />
            ))}
          </div>
        )}
      </section>

      <Card title={t("results.explanation.title")} padding="sm">
        <p className="text-sm text-muted">{t("results.explanation.placeholder")}</p>
      </Card>

      <ResultsActions />
    </div>
  );
}
