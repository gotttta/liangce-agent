import { useEffect, useMemo, useState } from "react";

// 响应式媒体查询（阶段 6 窄屏适配：≥ 1024px 用三栏，否则抽屉布局）。
// 值不一致时在渲染期调整（React 官方模式），变更事件里才走 setState。
export function useMediaQuery(query: string): boolean {
  const media = useMemo(
    () => (typeof window === "undefined" ? null : window.matchMedia(query)),
    [query],
  );
  const [matches, setMatches] = useState(() => media?.matches ?? false);

  if (media && matches !== media.matches) setMatches(media.matches);

  useEffect(() => {
    if (!media) return;
    const onChange = () => setMatches(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [media]);

  return matches;
}
