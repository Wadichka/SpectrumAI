import { Search, X } from "lucide-react";
import { useTranslation } from "react-i18next";

import type { FunctionalGroup } from "@/api/functionalGroups";
import Button from "@/components/ui/Button";
import Input from "@/components/ui/Input";

interface CompoundsFiltersProps {
  query: string;
  onQueryChange: (value: string) => void;
  groups: FunctionalGroup[];
  selectedCodes: ReadonlySet<string>;
  onToggleGroup: (code: string) => void;
  onReset: () => void;
}

export default function CompoundsFilters({
  query,
  onQueryChange,
  groups,
  selectedCodes,
  onToggleGroup,
  onReset,
}: CompoundsFiltersProps) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end gap-3">
        <div className="grow min-w-[260px]">
          <Input
            label={t("database.filters.search")}
            placeholder={t("database.filters.search")}
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            iconLeft={<Search className="size-4" aria-hidden="true" />}
            aria-label={t("database.filters.search")}
          />
        </div>
        <Button variant="tertiary" size="sm" onClick={onReset}>
          {t("database.filters.reset")}
        </Button>
      </div>
      <div>
        <p className="mb-2 text-sm font-medium text-ink">
          {t("database.filters.groups")}{" "}
          <span className="text-muted">
            ({t("database.filters.selected", { count: selectedCodes.size })})
          </span>
        </p>
        <div className="flex flex-wrap gap-2">
          {groups.map((group) => {
            const isActive = selectedCodes.has(group.code);
            return (
              <button
                key={group.code}
                type="button"
                onClick={() => onToggleGroup(group.code)}
                aria-pressed={isActive}
                className={[
                  "inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-medium border transition-colors",
                  isActive
                    ? "bg-primary text-white border-primary"
                    : "bg-surface text-ink border-line hover:bg-background",
                ].join(" ")}
                title={group.description ?? group.name}
              >
                {group.code} · {group.name}
                {isActive ? <X className="size-3" aria-hidden="true" /> : null}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
