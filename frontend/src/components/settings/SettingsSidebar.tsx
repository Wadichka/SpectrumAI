import { Beaker, Languages, Sliders } from "lucide-react";
import { useTranslation } from "react-i18next";

export type SettingsSection = "identification" | "interface" | "preprocessing";

interface SettingsSidebarProps {
  active: SettingsSection;
  onChange: (section: SettingsSection) => void;
}

const SECTIONS: { id: SettingsSection; icon: typeof Sliders }[] = [
  { id: "identification", icon: Sliders },
  { id: "interface", icon: Languages },
  { id: "preprocessing", icon: Beaker },
];

export default function SettingsSidebar({ active, onChange }: SettingsSidebarProps) {
  const { t } = useTranslation();
  return (
    <nav className="flex w-full shrink-0 flex-col gap-1 md:w-56" aria-label="settings sections">
      {SECTIONS.map(({ id, icon: Icon }) => {
        const isActive = id === active;
        return (
          <button
            key={id}
            type="button"
            onClick={() => onChange(id)}
            aria-current={isActive ? "true" : undefined}
            className={[
              "flex items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors",
              isActive
                ? "bg-primary-muted text-primary"
                : "text-ink hover:bg-background",
            ].join(" ")}
          >
            <Icon className="size-4" aria-hidden="true" />
            {t(`settings.sections.${id}`)}
          </button>
        );
      })}
    </nav>
  );
}
