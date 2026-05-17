import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Outlet } from "react-router-dom";

import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import { useSettingsStore } from "@/stores/useSettingsStore";

export default function AppLayout() {
  const { i18n } = useTranslation();
  const language = useSettingsStore((s) => s.language);

  useEffect(() => {
    if (i18n.language !== language) {
      void i18n.changeLanguage(language);
    }
  }, [i18n, language]);

  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-container px-6 py-8">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
