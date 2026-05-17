import { ExternalLink } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { structureSvgUrlBySmiles } from "@/api/compounds";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import type { components } from "@/api/types.gen";

type CompoundCandidate = components["schemas"]["CompoundCandidate"];

interface CandidateCardProps {
  candidate: CompoundCandidate;
}

export default function CandidateCard({ candidate }: CandidateCardProps) {
  const { t } = useTranslation();
  const displayName = candidate.name ?? candidate.formula ?? candidate.smiles;
  return (
    <Card padding="sm">
      <div className="flex gap-4">
        <img
          src={structureSvgUrlBySmiles(candidate.smiles, { width: 160, height: 120 })}
          alt={displayName}
          width={160}
          height={120}
          className="shrink-0 rounded border border-line bg-white"
        />
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex items-baseline justify-between gap-2">
            <h3 className="text-sm font-semibold text-ink">#{candidate.rank}. {displayName}</h3>
            <Badge variant={candidate.consistent ? "success" : "neutral"}>
              {t("results.candidate.score", { value: candidate.score.toFixed(3) })}
            </Badge>
          </div>
          {candidate.formula ? (
            <p className="text-xs text-muted">{candidate.formula}</p>
          ) : null}
          <p className="truncate font-mono text-xs text-muted" title={candidate.smiles}>
            {candidate.smiles}
          </p>
          {candidate.cas_number ? (
            <p className="text-xs text-muted">CAS: {candidate.cas_number}</p>
          ) : null}
          {candidate.consistent ? (
            <p className="text-xs text-success">
              {t("results.candidate.consistent", {
                jaccard: candidate.jaccard.toFixed(2),
              })}
            </p>
          ) : null}
          <div className="mt-1">
            <Link to={`/compounds/${candidate.compound_id}`}>
              <Button variant="tertiary" size="sm">
                <ExternalLink className="size-3" aria-hidden="true" />
                {t("results.candidate.more")}
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </Card>
  );
}
