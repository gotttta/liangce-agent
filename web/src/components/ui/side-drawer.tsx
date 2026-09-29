import { X } from "lucide-react";
import { useEffect, type ReactNode } from "react";
import { cn } from "@/lib/utils";

// 侧边抽屉（阶段 6 窄屏适配）：侧栏 / 产物面板在 < 1024px 下的容器。
// 不依赖新组件库：fixed 定位 + 背景遮罩 + Esc 关闭 + 焦点移入。

interface SideDrawerProps {
  open: boolean;
  onClose: () => void;
  side: "left" | "right";
  title: string;
  width?: number;
  children: ReactNode;
}

export function SideDrawer({ open, onClose, side, title, width = 320, children }: SideDrawerProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 lg:hidden" role="dialog" aria-modal="true" aria-label={title}>
      <button
        type="button"
        aria-label="关闭抽屉"
        className="absolute inset-0 h-full w-full cursor-default bg-foreground/25"
        onClick={onClose}
        tabIndex={-1}
      />
      <div
        className={cn(
          "absolute inset-y-0 flex w-[min(88vw,var(--drawer-width))] flex-col bg-background shadow-xl",
          side === "left" ? "left-0 border-r" : "right-0 border-l",
        )}
        style={{ "--drawer-width": `${width}px` } as React.CSSProperties}
      >
        <div className="flex h-11 shrink-0 items-center justify-between border-b px-3">
          <span className="text-sm font-medium">{title}</span>
          <button
            type="button"
            onClick={onClose}
            aria-label={`关闭${title}`}
            className="rounded-md p-1.5 hover:bg-muted"
          >
            <X className="size-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1">{children}</div>
      </div>
    </div>
  );
}
