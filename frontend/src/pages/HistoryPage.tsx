import { useTranslation } from "react-i18next";

import Card from "@/components/ui/Card";

export default function HistoryPage() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-2xl font-bold text-ink">{t("page.history.title")}</h1>
      <Card>
        <p className="text-muted">{t("page.history.stub")}</p>
      </Card>
    </div>
  );
}
