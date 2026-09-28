import { TransformComponent, TransformWrapper } from "react-zoom-pan-pinch";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";

// 大图查看器：支持缩放 / 拖动（react-zoom-pan-pinch，计划 §3 固定选型）
interface ImageViewerProps {
  src: string;
  alt: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ImageViewer({ src, alt, open, onOpenChange }: ImageViewerProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl sm:max-w-5xl">
        <DialogTitle className="text-sm font-medium">{alt}</DialogTitle>
        <TransformWrapper>
          <TransformComponent
            wrapperClass="!h-[70vh] w-full rounded-lg bg-muted/40"
            contentClass="flex items-center justify-center"
          >
            <img src={src} alt={alt} className="max-h-[68vh] select-none object-contain" />
          </TransformComponent>
        </TransformWrapper>
        <p className="text-xs text-muted-foreground">滚轮缩放，拖动平移</p>
      </DialogContent>
    </Dialog>
  );
}
