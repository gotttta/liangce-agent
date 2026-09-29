import { create } from "zustand";

// 客户端 UI 状态（计划 §3）：面板开合等；宽度由 react-resizable-panels 持久化。
// sidebarOpen 只在窄屏（< 1024px）下作为抽屉状态使用（计划 §9.1）。
interface UiState {
  rightPanelOpen: boolean;
  toggleRightPanel: () => void;
  setRightPanel: (open: boolean) => void;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

export const useUiStore = create<UiState>((set) => ({
  // 窄屏初次加载时产物面板是抽屉，默认收起（宽屏默认展开）
  rightPanelOpen:
    typeof window !== "undefined"
      ? window.matchMedia("(min-width: 1024px)").matches
      : true,
  toggleRightPanel: () => set((state) => ({ rightPanelOpen: !state.rightPanelOpen })),
  setRightPanel: (open) => set({ rightPanelOpen: open }),
  sidebarOpen: false,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
}));
