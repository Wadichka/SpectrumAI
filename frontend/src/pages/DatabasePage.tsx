import { Database } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { fetchCompounds, type PaginatedCompounds } from "@/api/compounds";
import { fetchFunctionalGroups, type FunctionalGroup } from "@/api/functionalGroups";
import CompoundsFilters from "@/components/database/CompoundsFilters";
import CompoundsTable from "@/components/database/CompoundsTable";
import Alert from "@/components/ui/Alert";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import Spinner from "@/components/ui/Spinner";

const PAGE_SIZE = 20;
const SEARCH_DEBOUNCE_MS = 300;

export default function DatabasePage() {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [groups, setGroups] = useState<FunctionalGroup[]>([]);
  const [selectedCodes, setSelectedCodes] = useState<Set<string>>(new Set());
  const [page, setPage] = useState(1);
  const [data, setData] = useState<PaginatedCompounds | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<number | null>(null);

  useEffect(() => {
    if (debounceRef.current != null) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => {
      setDebouncedQuery(query);
      setPage(1);
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      if (debounceRef.current != null) window.clearTimeout(debounceRef.current);
    };
  }, [query]);

  useEffect(() => {
    fetchFunctionalGroups()
      .then(setGroups)
      .catch(() => setGroups([]));
  }, []);

  const selectedCodesArray = useMemo(() => Array.from(selectedCodes), [selectedCodes]);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchCompounds({
        page,
        size: PAGE_SIZE,
        q: debouncedQuery.trim() || undefined,
        functional_groups: selectedCodesArray.length > 0 ? selectedCodesArray : undefined,
      });
      setData(result);
    } catch {
      setError(t("database.load_error"));
    } finally {
      setLoading(false);
    }
  }, [page, debouncedQuery, selectedCodesArray, t]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const handleToggleGroup = (code: string) => {
    setSelectedCodes((prev) => {
      const next = new Set(prev);
      if (next.has(code)) next.delete(code);
      else next.add(code);
      return next;
    });
    setPage(1);
  };

  const handleReset = () => {
    setQuery("");
    setSelectedCodes(new Set());
    setPage(1);
  };

  const rows = data?.data ?? [];
  const total = data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-bold text-ink">{t("database.title")}</h1>
        <Button
          variant="tertiary"
          onClick={() => window.alert(t("database.add_stub"))}
        >
          {t("database.add")}
        </Button>
      </div>

      <p className="text-sm text-muted">
        {t("database.stats.filtered", { count: total })}
      </p>

      <Card>
        <CompoundsFilters
          query={query}
          onQueryChange={setQuery}
          groups={groups}
          selectedCodes={selectedCodes}
          onToggleGroup={handleToggleGroup}
          onReset={handleReset}
        />
      </Card>

      {error ? <Alert variant="error">{error}</Alert> : null}

      <Card padding="sm">
        {loading ? (
          <div className="flex items-center justify-center py-10">
            <Spinner />
          </div>
        ) : rows.length === 0 ? (
          <div className="flex flex-col items-center gap-3 py-10 text-center">
            <Database className="size-10 text-muted" aria-hidden="true" />
            <p className="text-base font-semibold text-ink">{t("database.empty.title")}</p>
            <p className="text-sm text-muted">{t("database.empty.description")}</p>
          </div>
        ) : (
          <CompoundsTable rows={rows} />
        )}

        {rows.length > 0 ? (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3 text-sm">
            <span className="text-muted">
              {t("history.pagination.summary", { page, pages, total })}
            </span>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="tertiary"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
              >
                {t("history.pagination.prev")}
              </Button>
              <Button
                size="sm"
                variant="tertiary"
                onClick={() => setPage((p) => Math.min(pages, p + 1))}
                disabled={page >= pages}
              >
                {t("history.pagination.next")}
              </Button>
            </div>
          </div>
        ) : null}
      </Card>
    </div>
  );
}
