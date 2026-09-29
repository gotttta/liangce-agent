import { CircleAlert, CircleCheck } from "lucide-react";
import { create } from "zustand";
import { cn } from "@/lib/utils";

// 轻量 toast（阶段 6）：不引入新依赖，zustand 栈 + 固定定位渲染。
// 用于 ReviewCard 409“任务仍在运行”等瞬态错误提示（计划 §9.4）。

export interface ToastItem {
  id: number;
  message: string;
  variant: "info" | "error";
}

interface ToastStore {
  toasts: ToastItem[];
  push: (message: string, variant?: ToastItem["variant"]) => void;
  dismiss: (id: number) => void;
}

let nextToastId = 1;

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  push: (message, variant = "info") => {
    const id = nextToastId;
    nextToastId += 1;
    set((state) => ({ toasts: [...state.toasts, { id, message, variant }] }));
    window.setTimeout(() => {
      set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) }));
    }, 4000);
  },
  dismiss: (id) => set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) })),
}));

export function toast(message: string, variant: ToastItem["variant"] = "info") {
  useToastStore.getState().push(message, variant);
}

export function Toaster() {
  const toasts = useToastStore((state) => state.toasts);
  const dismiss = useToastStore((state) => state.dismiss);
  if (toasts.length === 0) return null;
  return (
    <div className="pointer-events-none fixed bottom-6 left-1/2 z-100 flex w-full max-w-sm -translate-x-1/2 flex-col gap-2 px-4">
      {toasts.map((item) => (
        <button
          key={item.id}
          type="button"
          onClick={() => dismiss(item.id)}
          className={cn(
            "pointer-events-auto flex items-center gap-2 rounded-xl border bg-background px-3.5 py-2.5 text-left text-sm shadow-lg",
            item.variant === "error"
              ? "border-red-300 text-red-700 dark:border-red-900/70 dark:text-red-400"
              : "border-border",
          )}
          aria-label={`关闭提示：${item.message}`}
        >
          {item.variant === "error" ? (
            <CircleAlert className="size-4 shrink-0" />
          ) : (
            <CircleCheck className="size-4 shrink-0 text-emerald-600" />
          )}
          <span className="min-w-0 flex-1 break-words">{item.message}</span>
        </button>
      ))}
    </div>
  );
}
