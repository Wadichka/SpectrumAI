import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useParams } from "react-router-dom";

import { fetchHistoryDetail } from "@/api/history";
import CandidateCard from "@/components/results/CandidateCard";
import FunctionalGroupBadges from "@/components/results/FunctionalGroupBadges";
import ResultsActions from "@/components/results/ResultsActions";
import SpectrumChart from "@/components/results/SpectrumChart";
import Alert from "@/components/ui/Alert";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import Spinner from "@/components/ui/Spinner";
import { useIdentificationStore } from "@/stores/useIdentificationStore";

export default function ResultsPage() {
  const { t } = useTranslation();
  const { requestId: requestIdParam } = useParams<{ requestId?: string }>();
  const response = useIdentificationStore((s) => s.lastResponse);
  const lastRequestId = useIdentificationStore((s) => s.lastRequestId);
  const setLastResponse = useIdentificationStore((s) => s.setLastResponse);
  const setLastRequestId = useIdentificationStore((s) => s.setLastRequestId);

  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedCode, setSelectedCode] = useState<string | null>(
    response?.gradcam?.group_code ?? null,
  );

  // Если в URL есть :requestId, который не совпадает с тем, что в сторе —
  // подгружаем полный сохранённый ответ /identify из истории. Это закрывает
  // баг: до фикса ResultsPage всегда показывал «последний live»-результат,
  // независимо от того, какую запись юзер открыл из /history.
  useEffect(() => {
    if (!requestIdParam) return;
    const requestId = Number(requestIdParam);
    if (!Number.isFinite(requestId) || requestId <= 0) return;
    if (lastRequestId === requestId && response) return;

    const controller = new AbortController();
    setLoading(true);
    setLoadError(null);
    fetchHistoryDetail(requestId, controller.signal)
      .then((detail) => {
        setLastResponse(detail);
        setLastRequestId(requestId);
        setSelectedCode(detail.gradcam?.group_code ?? null);
      })
      .catch((err: unknown) => {
        if ((err as Error)?.name === "CanceledError") return;
        setLoadError(t("results.lost_session"));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [requestIdParam, lastRequestId, response, setLastResponse, setLastRequestId, t]);

  const sortedCandidates = useMemo(
    () => (response ? [...response.candidates].sort((a, b) => a.rank - b.rank) : []),
    [response],
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Spinner />
      </div>
    );
  }

  if (!response || loadError) {
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
        rawSpectrum={response.raw_spectrum ?? null}
        rawWavenumbers={response.raw_wavenumbers ?? null}
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
