import { Database, FlaskConical, History, Layers, Settings, Sparkles } from "lucide-react";
import type { ComponentType } from "react";
import { useTranslation } from "react-i18next";
import { NavLink } from "react-router-dom";

import { cn } from "@/lib/cn";

interface NavItem {
  to: string;
  labelKey: string;
  icon: ComponentType<{ className?: string; "aria-hidden"?: boolean | "true" | "false" }>;
}

const ITEMS: NavItem[] = [
  { to: "/identify", labelKey: "nav.identify", icon: Sparkles },
  { to: "/batch", labelKey: "nav.batch", icon: Layers },
  { to: "/compounds", labelKey: "nav.database", icon: Database },
  { to: "/history", labelKey: "nav.history", icon: History },
  { to: "/settings", labelKey: "nav.settings", icon: Settings },
];

export default function Sidebar() {
  const { t } = useTranslation();
  return (
    <aside
      aria-label={t("nav.identify")}
      className="hidden w-60 shrink-0 border-r border-line bg-surface md:flex md:flex-col"
    >
      <div className="flex h-16 items-center gap-2 border-b border-line px-6 text-primary">
        <FlaskConical className="size-6" aria-hidden="true" />
        <span className="text-base font-semibold">{t("app.title")}</span>
      </div>
      <nav className="flex flex-1 flex-col gap-1 p-4">
        {ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary-muted text-primary"
                    : "text-muted hover:bg-background hover:text-ink",
                )
              }
            >
              <Icon className="size-4" aria-hidden="true" />
              <span>{t(item.labelKey)}</span>
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
}
