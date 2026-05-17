import { Database, Download, Search, Share2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

import Button from "@/components/ui/Button";

export default function ResultsActions() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const stub = () => window.alert(t("results.actions.stub"));
  return (
    <div className="flex flex-wrap items-center justify-end gap-2">
      <Button variant="primary" onClick={stub}>
        <Download className="size-4" aria-hidden="true" />
        {t("results.actions.export")}
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
