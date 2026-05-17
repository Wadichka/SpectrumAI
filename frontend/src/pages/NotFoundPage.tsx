import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";

export default function NotFoundPage() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-4">
      <Card title={t("page.not_found.title")}>
        <p className="mb-4 text-muted">{t("page.not_found.description")}</p>
        <Link to="/identify">
          <Button variant="tertiary">{t("nav.identify")}</Button>
        </Link>
      </Card>
    </div>
  );
}
