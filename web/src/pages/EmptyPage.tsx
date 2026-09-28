export function EmptyPage() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 p-8 text-center">
      <h1 className="text-lg font-semibold">两策视觉代理</h1>
      <p className="max-w-md text-sm text-muted-foreground">
        从左侧选择一个任务，或新建任务开始。上传样本图、描述检测目标，
        代理会生成参考掩膜并迭代算子流水线。
      </p>
    </div>
  );
}
