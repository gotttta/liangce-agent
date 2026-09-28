import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router";
import { api } from "@/api/client";
import { TopBar } from "@/components/layout/TopBar";
import { formatTimestamp } from "@/lib/format";

// 阶段 5 接入时间线与 Composer 前的任务页骨架：标题栏 + 样本图预览
export function TaskPage() {
  const { taskId } = useParams();
  const task = useQuery({
    queryKey: ["task", taskId],
    queryFn: () => api.getTask(taskId!),
    enabled: Boolean(taskId),
    refetchInterval: 5_000,
  });

  if (task.isPending) {
    return <div className="p-8 text-sm text-muted-foreground">加载任务…</div>;
  }
  if (task.isError || !task.data) {
    return (
      <div className="p-8 text-sm text-red-500">
        任务加载失败：{task.error instanceof Error ? task.error.message : "未知错误"}
      </div>
    );
  }

  const detail = task.data;
  return (
    <div className="flex h-full min-h-0 flex-col">
      <TopBar task={detail} />
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-6 py-8">
          <p className="text-sm text-muted-foreground">
            创建于 {formatTimestamp(detail.created_at)} · 样本 {detail.sample_count} 张
          </p>
          {detail.samples.length > 0 ? (
            <section className="mt-6">
              <h2 className="mb-2 text-sm font-medium">样本图</h2>
              <div className="flex flex-wrap gap-3">
                {detail.samples.map((sample) => (
                  <a
                    key={sample.name}
                    href={sample.url}
                    target="_blank"
                    rel="noreferrer"
                    className="block overflow-hidden rounded-lg border bg-background"
                    title={sample.name}
                  >
                    <img
                      src={sample.url}
                      alt={sample.name}
                      className="size-24 object-cover"
                      loading="lazy"
                    />
                  </a>
                ))}
              </div>
            </section>
          ) : (
            <p className="mt-8 rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">
              上传样本图并描述检测目标后开始（输入区将在下一步提供）
            </p>
          )}
        </div>
      </main>
    </div>
  );
}
