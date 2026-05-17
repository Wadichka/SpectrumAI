import { AxiosError } from "axios";
import { ArrowLeft } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useParams } from "react-router-dom";

import { fetchCompoundDetail, structureSvgUrlBySmiles, type CompoundDetail } from "@/api/compounds";
import Alert from "@/components/ui/Alert";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import Spinner from "@/components/ui/Spinner";
import { useIdentificationStore } from "@/stores/useIdentificationStore";

interface CardData {
  name: string | null;
  formula: string | null;
  cas_number: string | null;
  smiles: string;
  inchi: string | null;
  inchi_key: string | null;
  molecular_weight: number | null;
  functional_groups: string[];
}

function fromDetail(detail: CompoundDetail): CardData {
  return {
    name: detail.name,
    formula: detail.formula,
    cas_number: detail.cas_number,
    smiles: detail.smiles,
    inchi: detail.inchi,
    inchi_key: detail.inchi_key,
    molecular_weight: detail.molecular_weight,
    functional_groups: detail.functional_groups,
  };
}

export default function CompoundCardPage() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const compoundId = id ? Number(id) : Number.NaN;
  const lastResponse = useIdentificationStore((s) => s.lastResponse);
  const [data, setData] = useState<CardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!Number.isFinite(compoundId)) {
      setLoading(false);
      return;
    }
    const fallback = lastResponse?.candidates.find((c) => c.compound_id === compoundId);
    setLoading(true);
    fetchCompoundDetail(compoundId)
      .then((detail) => {
        if (!cancelled) {
          setData(fromDetail(detail));
          setErrorMessage(null);
        }
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const status = error instanceof AxiosError ? error.response?.status : undefined;
        if (fallback) {
          setData({
            name: fallback.name ?? null,
            formula: fallback.formula ?? null,
            cas_number: fallback.cas_number ?? null,
            smiles: fallback.smiles,
            inchi: null,
            inchi_key: null,
            molecular_weight: null,
            functional_groups: fallback.matched_groups.slice(),
          });
          setErrorMessage(null);
          // status проверяется только для логирования — fallback одинаков и для 404, и для прочих ошибок.
          void status;
        } else {
          setErrorMessage(t("compound.not_found"));
          setData(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [compoundId, lastResponse, t]);

  return (
    <div className="flex flex-col gap-4">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-ink">{t("compound.title")}</h1>
        <Link to="/identify/results">
          <Button variant="tertiary">
            <ArrowLeft className="size-4" aria-hidden="true" />
            {t("compound.back")}
          </Button>
        </Link>
      </header>

      {loading ? (
        <div className="flex items-center gap-2 text-sm text-muted">
          <Spinner size="sm" />
          <span>{t("common.loading")}</span>
        </div>
      ) : errorMessage ? (
        <Alert variant="warning">{errorMessage}</Alert>
      ) : data ? (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_1fr]">
          <Card padding="sm">
            <img
              src={structureSvgUrlBySmiles(data.smiles, { width: 280, height: 220 })}
              alt={data.name ?? data.smiles}
              width={280}
              height={220}
              className="mx-auto rounded border border-line bg-white"
            />
            <div className="mt-3 flex flex-wrap gap-1">
              {data.functional_groups.map((code) => (
                <Badge key={code} variant="info">
                  {code}
                </Badge>
              ))}
            </div>
          </Card>

          <Card padding="sm">
            <dl className="grid grid-cols-1 gap-2 text-sm">
              <Field label={t("compound.field.name")} value={data.name} />
              <Field label={t("compound.field.formula")} value={data.formula} />
              <Field label={t("compound.field.cas")} value={data.cas_number} />
              <Field label={t("compound.field.smiles")} value={data.smiles} mono />
              <Field label={t("compound.field.inchi")} value={data.inchi} mono />
              <Field label={t("compound.field.inchi_key")} value={data.inchi_key} mono />
              <Field
                label={t("compound.field.molecular_weight")}
                value={
                  data.molecular_weight !== null
                    ? `${data.molecular_weight.toFixed(2)} г/моль`
                    : null
                }
              />
            </dl>
          </Card>
        </div>
      ) : (
        <Alert variant="warning">{t("compound.not_found")}</Alert>
      )}
    </div>
  );
}

interface FieldProps {
  label: string;
  value: string | null;
  mono?: boolean;
}

function Field({ label, value, mono = false }: FieldProps) {
  return (
    <div className="grid grid-cols-[140px_1fr] gap-3">
      <dt className="text-xs uppercase tracking-wide text-muted">{label}</dt>
      <dd
        className={
          mono
            ? "break-all font-mono text-sm text-ink"
            : "text-sm text-ink"
        }
      >
        {value ?? "—"}
      </dd>
    </div>
  );
}
