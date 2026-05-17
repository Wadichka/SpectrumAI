import { useTranslation } from "react-i18next";

import Card from "@/components/ui/Card";

export default function IdentificationPage() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-2xl font-bold text-ink">{t("page.identify.title")}</h1>
      <Card>
        <p className="text-muted">{t("page.identify.stub")}</p>
      </Card>
    </div>
  );
}
