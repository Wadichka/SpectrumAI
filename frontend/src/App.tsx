import { Navigate, Route, Routes } from "react-router-dom";

import AppLayout from "@/components/layout/AppLayout";
import BatchPage from "@/pages/BatchPage";
import CompoundCardPage from "@/pages/CompoundCardPage";
import DatabasePage from "@/pages/DatabasePage";
import HistoryPage from "@/pages/HistoryPage";
import IdentificationPage from "@/pages/IdentificationPage";
import NotFoundPage from "@/pages/NotFoundPage";
import ResultsPage from "@/pages/ResultsPage";
import SettingsPage from "@/pages/SettingsPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to="/identify" replace />} />
        <Route path="/identify" element={<IdentificationPage />} />
        <Route path="/identify/results/:requestId?" element={<ResultsPage />} />
        <Route path="/batch" element={<BatchPage />} />
        <Route path="/compounds" element={<DatabasePage />} />
        <Route path="/compounds/:id" element={<CompoundCardPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
