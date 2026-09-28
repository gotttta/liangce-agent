import { Loader2, X } from "lucide-react";
import type { SampleOut } from "@/api/types";

interface AttachmentChipsProps {
  samples: SampleOut[];
  uploading: string[];
  onRemove: (name: string) => void;
}

// 样本图缩略图 chip：上传完成后立即出现在输入框上方，可删除（计划 §9.3）
export function AttachmentChips({ samples, uploading, onRemove }: AttachmentChipsProps) {
  if (samples.length === 0 && uploading.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-2 px-1 pb-2">
      {samples.map((sample) => (
        <div
          key={sample.name}
          className="group relative size-16 overflow-hidden rounded-lg border bg-background"
          title={sample.name}
        >
          <img
            src={sample.url}
            alt={sample.name}
            className="size-full object-cover"
            loading="lazy"
          />
          <button
            type="button"
            onClick={() => onRemove(sample.name)}
            aria-label={`删除样本图 ${sample.name}`}
            className="absolute top-0.5 right-0.5 hidden rounded bg-background/80 p-0.5 group-hover:block"
          >
            <X className="size-3" />
          </button>
        </div>
      ))}
      {uploading.map((name) => (
        <div
          key={name}
          className="flex size-16 items-center justify-center rounded-lg border border-dashed"
          title={`上传中：${name}`}
        >
          <Loader2 className="size-4 animate-spin text-muted-foreground" />
        </div>
      ))}
    </div>
  );
}
