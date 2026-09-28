// 流式思考文本：等宽小字、灰色、最多 12 行可滚动（计划 §9.2）。
// 追加文本只改动这个节点的文本内容，key 稳定不会整体重挂载导致闪烁。
export function ThinkingBlock({ text }: { text: string }) {
  if (!text) return null;
  return (
    <div className="max-h-[15rem] overflow-y-auto rounded-lg bg-muted/50 px-3 py-2 font-mono text-xs leading-5 whitespace-pre-wrap break-words text-muted-foreground">
      {text}
    </div>
  );
}
