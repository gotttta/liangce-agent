import { Eye, EyeOff } from "lucide-react";
import { useMemo, useState } from "react";
import type { ReferenceInfo, SampleOut } from "@/api/types";
import { formatScore } from "@/lib/format";
import { cn } from "@/lib/utils";
import { useTimelineStore } from "@/store/timeline";
import { EmptyHint } from "./ArtifactPanel";

// 参考标签页（计划 §9.6）：所有样本图网格；原图 / 叠加图切换、
// 状态（未生成 / 待确认 / 已确认 / 已跳过打分）、SAM 分数。
// 已确认的参考来自 GET /artifacts；待确认的候选来自时间线事件（还没落盘）。

type ReferenceStatus = "pending" | "confirmed" | "skipped" | "none";

interface GalleryEntry {
  imageId: string;
  imageUrl: string;
  overlayUrl: string | null;
  samScore: number | null;
  status: ReferenceStatus;
}

function statusLabel(status: ReferenceStatus): string {
  switch (status) {
    case "confirmed":
      return "已确认";
    case "skipped":
      return "已跳过打分";
    case "pending":
      return "待确认";
    default:
      return "未生成";
  }
}

function statusClass(status: ReferenceStatus): string {
  switch (status) {
    case "confirmed":
      return "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400";
    case "skipped":
      return "bg-muted text-muted-foreground";
    case "pending":
      return "bg-amber-100 text-amber-700 dark:bg-amber-950/60 dark:text-amber-400";
    default:
      return "bg-muted text-muted-foreground/70";
  }
}

function buildEntries(samples: SampleOut[], references: ReferenceInfo[]): GalleryEntry[] {
  const confirmed = new Map(
    references.map((reference) => [reference.image_id, reference] as const),
  );
  const entries: GalleryEntry[] = samples.map((sample) => {
    const imageId = sample.name.replace(/\.[^.]+$/, "");
    const reference = confirmed.get(imageId);
    if (reference) {
      return {
        imageId,
        imageUrl: reference.image_url || sample.url,
        overlayUrl: reference.overlay_url,
        samScore: reference.sam_iou_score,
        status: reference.skip_scoring ? "skipped" : "confirmed",
      };
    }
    return { imageId, imageUrl: sample.url, overlayUrl: null, samScore: null, status: "none" };
  });
  // artifacts 里可能出现已删除样本的参考（按 image_id 兜底补到网格尾部）
  const known = new Set(entries.map((entry) => entry.imageId));
  for (const reference of references) {
    if (known.has(reference.image_id)) continue;
    entries.push({
      imageId: reference.image_id,
      imageUrl: reference.image_url,
      overlayUrl: reference.overlay_url,
      samScore: reference.sam_iou_score,
      status: reference.skip_scoring ? "skipped" : "confirmed",
    });
  }
  return entries;
}

function ReferenceTile({ entry }: { entry: GalleryEntry }) {
  const [showOverlay, setShowOverlay] = useState(Boolean(entry.overlayUrl));
  const effectiveOverlay = showOverlay ? entry.overlayUrl : null;

  return (
    <div className="overflow-hidden rounded-xl border bg-background">
      <div className="relative aspect-4/3 bg-muted/40">
        <img
          src={effectiveOverlay ?? entry.imageUrl}
          alt={`${entry.imageId}${effectiveOverlay ? " 参考掩膜叠加图" : " 样本图"}`}
          className="size-full object-contain"
          loading="lazy"
        />
        {entry.overlayUrl && (
          <button
            type="button"
            onClick={() => setShowOverlay(!showOverlay)}
            aria-label={showOverlay ? "切换为原图" : "切换为叠加图"}
            title={showOverlay ? "查看原图" : "查看叠加图"}
            className="absolute top-1.5 right-1.5 rounded-md bg-background/85 p-1.5 shadow-sm hover:bg-background"
          >
            {showOverlay ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
          </button>
        )}
      </div>
      <div className="flex items-center gap-2 px-2.5 py-2 text-xs">
        <span className="min-w-0 flex-1 truncate font-mono" title={entry.imageId}>
          {entry.imageId}
        </span>
        <span className={cn("shrink-0 rounded-full px-1.5 py-0.5 text-[10px]", statusClass(entry.status))}>
          {statusLabel(entry.status)}
        </span>
      </div>
      {entry.samScore !== null && (
        <div className="px-2.5 pb-2 text-[11px] text-muted-foreground">
          SAM {formatScore(entry.samScore)}
        </div>
      )}
    </div>
  );
}

export function ReferenceGallery({
  samples,
  references,
}: {
  samples: SampleOut[];
  references: ReferenceInfo[];
}) {
  // 待确认的候选：当前任务时间线里最新一条该图的参考事件。
  // selector 只取稳定引用（items），过滤放 useMemo——派生数组放进
  // selector 会造成快照不稳定，触发 useSyncExternalStore 无限重渲染。
  const items = useTimelineStore((state) => state.items);
  const timelineReferences = useMemo(
    () => items.filter((item) => item.kind === "reference"),
    [items],
  );
  const pendingById = useMemo(() => {
    const map = new Map<string, { overlayUrl: string | null; samScore: number }>();
    for (const item of timelineReferences) {
      if (item.kind !== "reference") continue;
      map.set(item.imageId, { overlayUrl: item.overlayUrl, samScore: item.samScore });
    }
    return map;
  }, [timelineReferences]);

  const entries = buildEntries(samples, references).map((entry) => {
    if (entry.status !== "none" || !pendingById.has(entry.imageId)) return entry;
    const pending = pendingById.get(entry.imageId)!;
    return {
      ...entry,
      overlayUrl: entry.overlayUrl ?? pending.overlayUrl,
      samScore: pending.samScore,
      status: "pending" as ReferenceStatus,
    };
  });

  if (entries.length === 0) {
    return <EmptyHint text="还没有样本图；上传后这里会显示参考掩膜状态。" />;
  }
  return (
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-3">
      {entries.map((entry) => (
        <ReferenceTile key={entry.imageId} entry={entry} />
      ))}
    </div>
  );
}
