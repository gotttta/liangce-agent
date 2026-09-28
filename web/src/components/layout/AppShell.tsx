import {
  Group,
  Panel,
  Separator,
  type Layout,
} from "react-resizable-panels";
import { useCallback, useState } from "react";
import { Outlet } from "react-router";
import { useUiStore } from "@/store/ui";
import { Sidebar } from "./Sidebar";

// 三栏布局：侧栏 / 主区 / 产物面板；宽度百分比存 localStorage（计划 §9.1）
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
  const rightPanelOpen = useUiStore((state) => state.rightPanelOpen);
  const [defaultLayout] = useState(readSavedLayout);

  const saveLayout = useCallback((layout: Layout) => {
    try {
      localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(layout));
    } catch {
      // 隐私模式等场景下 localStorage 不可用：布局持久化是可选能力
    }
  }, []);

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
            <div className="flex h-full flex-col">
              <div className="border-b px-4 py-3 text-sm font-medium text-muted-foreground">
                产物
              </div>
              <div className="flex flex-1 items-center justify-center p-6 text-sm text-muted-foreground">
                参考掩膜、迭代分数与最优算法将在这里展示
              </div>
            </div>
          </Panel>
        </>
      )}
    </Group>
  );
}
