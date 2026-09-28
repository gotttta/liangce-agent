import { ScanSearch } from "lucide-react";
import { useState } from "react";
import { formatScore } from "@/lib/format";
import type { ReferenceItem } from "@/store/timeline";
import { ImageViewer } from "./ImageViewer";

// 参考掩膜卡片：叠加图缩略图（点击放大）、image_id、SAM 分数；
// 低质量时提示确认后将不参与打分（计划 §9.2）
export function ReferenceCard({ reference }: { reference: ReferenceItem }) {
  const [viewerOpen, setViewerOpen] = useState(false);

  return (
    <div className="rounded-xl border p-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
        <ScanSearch className="size-4 text-muted-foreground" />
        <span className="font-medium">参考掩膜 · {reference.imageId}</span>
        <span className="text-xs text-muted-foreground">
          SAM 分数 {formatScore(reference.samScore)}
        </span>
      </div>
      {reference.lowQuality && (
        <p className="mt-2 rounded-lg bg-amber-50 px-3 py-1.5 text-xs text-amber-700 dark:bg-amber-950/40 dark:text-amber-400">
          分割质量存疑，确认后将不参与打分
        </p>
      )}
      {reference.overlayUrl && (
        <button
          type="button"
          onClick={() => setViewerOpen(true)}
          className="mt-2 block overflow-hidden rounded-lg border hover:opacity-90"
          aria-label={`查看 ${reference.imageId} 叠加图大图`}
        >
          <img
            src={reference.overlayUrl}
            alt={`${reference.imageId} 参考掩膜叠加图`}
            className="max-h-64 w-auto object-contain"
            loading="lazy"
          />
        </button>
      )}
      {reference.overlayUrl && (
        <ImageViewer
          src={reference.overlayUrl}
          alt={`${reference.imageId} 参考掩膜叠加图`}
          open={viewerOpen}
          onOpenChange={setViewerOpen}
        />
      )}
    </div>
  );
}
