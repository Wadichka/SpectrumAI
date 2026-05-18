import { Database, Download, Search, Share2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

import { postIdentificationReport } from "@/api/reports";
import Button from "@/components/ui/Button";
import { useIdentificationStore } from "@/stores/useIdentificationStore";

export default function ResultsActions() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const response = useIdentificationStore((s) => s.lastResponse);
  const [exporting, setExporting] = useState(false);
  const stub = () => window.alert(t("results.actions.stub"));

  const handleExport = async () => {
    if (!response) {
      window.alert(t("results.actions.export_missing_data"));
      return;
    }
    setExporting(true);
    try {
      const blob = await postIdentificationReport(response);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const suffix = response.request_id ?? "unsaved";
      link.download = `identification-${suffix}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("PDF export failed", err);
      window.alert(t("results.actions.export_failed"));
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center justify-end gap-2">
      <Button variant="primary" onClick={handleExport} disabled={exporting || !response}>
        <Download className="size-4" aria-hidden="true" />
        {exporting ? t("results.actions.export_progress") : t("results.actions.export")}
      </Button>
      <Button variant="secondary" onClick={stub}>
        <Share2 className="size-4" aria-hidden="true" />
        {t("results.actions.share")}
      </Button>
      <Button variant="secondary" onClick={stub}>
        <Database className="size-4" aria-hidden="true" />
        {t("results.actions.save_to_db")}
      </Button>
      <Button variant="tertiary" onClick={() => navigate("/identify")}>
        <Search className="size-4" aria-hidden="true" />
        {t("results.actions.new_search")}
      </Button>
    </div>
  );
}
