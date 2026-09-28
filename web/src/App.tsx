import { Route, Routes } from "react-router";
import { AppShell } from "@/components/layout/AppShell";
import { EmptyPage } from "@/pages/EmptyPage";
import { TaskPage } from "@/pages/TaskPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<EmptyPage />} />
        <Route path="tasks/:taskId" element={<TaskPage />} />
      </Route>
    </Routes>
  );
}
