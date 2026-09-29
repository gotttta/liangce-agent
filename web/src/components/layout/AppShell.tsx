import { useQueryClient } from "@tanstack/react-query";
import {
  Group,
  Panel,
  Separator,
  type Layout,
} from "react-resizable-panels";
import { useCallback, useEffect, useState } from "react";
import { Outlet, useNavigate } from "react-router";
import { api } from "@/api/client";
import { ArtifactPanel } from "@/components/artifacts/ArtifactPanel";
import { SideDrawer } from "@/components/ui/side-drawer";
import { useMediaQuery } from "@/lib/hooks";
import { useUiStore } from "@/store/ui";
import { Sidebar } from "./Sidebar";

// 三栏布局：侧栏 / 主区 / 产物面板；宽度百分比存 localStorage（计划 §9.1）。
// 窄屏（< 1024px）改用抽屉：侧栏从左滑出，产物面板从右滑出（计划 §9.1）。
const LAYOUT_STORAGE_KEY = "liangce.shell-layout";

function readSavedLayout(): Layout | undefined {
  try {
    const raw = localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (!raw) return undefined;
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const layout: Layout = {};
    for (const [id, value] of Object.entries(parsed)) {
      if (typeof value === "number" && Number.isFinite(value)) layout[id] = value;
    }
    return Object.keys(layout).length > 0 ? layout : undefined;
  } catch {
    return undefined;
  }
}

export function AppShell() {
  const wide = useMediaQuery("(min-width: 1024px)");
  const rightPanelOpen = useUiStore((state) => state.rightPanelOpen);
  const toggleRightPanel = useUiStore((state) => state.toggleRightPanel);
  const sidebarOpen = useUiStore((state) => state.sidebarOpen);
  const setSidebarOpen = useUiStore((state) => state.setSidebarOpen);
  const setRightPanel = useUiStore((state) => state.setRightPanel);
  const [defaultLayout] = useState(readSavedLayout);
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const saveLayout = useCallback((layout: Layout) => {
    try {
      localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(layout));
    } catch {
      // 隐私模式等场景下 localStorage 不可用：布局持久化是可选能力
    }
  }, []);

  // 键盘快捷键（计划 §阶段6）：Cmd/Ctrl+K 新建任务，Cmd/Ctrl+. 切换产物面板
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const mod = event.metaKey || event.ctrlKey;
      if (!mod) return;
      if (event.repeat) return; // 按住不放只触发一次
      if (event.key.toLowerCase() === "k") {
        event.preventDefault();
        void api
          .createTask()
          .then((task) => {
            void queryClient.invalidateQueries({ queryKey: ["tasks"] });
            navigate(`/tasks/${task.id}`);
          })
          .catch(() => {
            // 创建失败不弹窗：侧栏下次刷新会重试列表
          });
      }
      if (event.key === ".") {
        event.preventDefault();
        toggleRightPanel();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate, queryClient, toggleRightPanel]);

  if (!wide) {
    return (
      <div className="h-svh bg-background text-foreground [overflow-anchor:none]">
        <Outlet />
        <SideDrawer
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          side="left"
          title="任务"
          width={300}
        >
          <Sidebar />
        </SideDrawer>
        <SideDrawer
          open={rightPanelOpen}
          onClose={() => setRightPanel(false)}
          side="right"
          title="产物"
          width={380}
        >
          <ArtifactPanel />
        </SideDrawer>
      </div>
    );
  }

  return (
    <Group
      orientation="horizontal"
      className="h-svh bg-background text-foreground"
      defaultLayout={defaultLayout}
      onLayoutChanged={saveLayout}
    >
      <Panel id="sidebar" defaultSize={260} minSize={200} maxSize={360} className="min-w-0">
        <Sidebar />
      </Panel>
      <Separator className="w-px bg-border" />
      <Panel id="main" minSize="40" className="min-w-0">
        <Outlet />
      </Panel>
      {rightPanelOpen && (
        <>
          <Separator className="w-px bg-border" />
          <Panel
            id="artifacts"
            defaultSize={380}
            minSize={280}
            maxSize="45%"
            className="min-w-0 bg-muted/30"
          >
            <ArtifactPanel />
          </Panel>
        </>
      )}
    </Group>
  );
}
