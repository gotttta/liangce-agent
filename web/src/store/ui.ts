import { create } from "zustand";

// 客户端 UI 状态（计划 §3）：面板开合等；宽度由 react-resizable-panels 持久化
interface UiState {
  rightPanelOpen: boolean;
  toggleRightPanel: () => void;
  setRightPanel: (open: boolean) => void;
}

export const useUiStore = create<UiState>((set) => ({
  rightPanelOpen: true,
  toggleRightPanel: () => set((state) => ({ rightPanelOpen: !state.rightPanelOpen })),
  setRightPanel: (open) => set({ rightPanelOpen: open }),
}));
