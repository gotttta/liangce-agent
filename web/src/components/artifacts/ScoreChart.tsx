import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { IterationRecord } from "@/api/types";
import { formatScore } from "@/lib/format";
import { EmptyHint } from "./ArtifactPanel";

// 迭代标签页（计划 §9.6）：composite_mean 折线 + 历史最优阶梯线，
// 下面是最优一轮的每图分数表。
function bestIteration(iterations: IterationRecord[]): IterationRecord | null {
  if (iterations.length === 0) return null;
  return iterations.reduce((best, record) =>
    record.run_score.composite_mean > best.run_score.composite_mean ? record : best,
  );
}

interface ChartPoint {
  iteration: number;
  score: number;
  best: number;
}

export function ScoreChart({ iterations }: { iterations: IterationRecord[] }) {
  if (iterations.length === 0) {
    return <EmptyHint text="还没有迭代记录；进入迭代阶段后这里会显示分数曲线。" />;
  }

  // core 的 iteration_history.jsonl 按行计数从 0 开始，而事件流里的
  // iteration_scored 从 1 开始；展示层 +1 对齐时间线的“第 N 轮”（见 issues 文档）
  const points = iterations.reduce<ChartPoint[]>((acc, record) => {
    const previous = acc.length > 0 ? acc[acc.length - 1].best : 0;
    const score = record.run_score.composite_mean;
    acc.push({
      iteration: record.iteration + 1,
      score,
      best: Math.max(previous, score),
    });
    return acc;
  }, []);
  const overallBest = points.length > 0 ? points[points.length - 1].best : 0;
  const bestRecord = bestIteration(iterations);

  return (
    <div className="space-y-4">
      <div
        className="h-56 w-full"
        role="img"
        aria-label={`迭代分数曲线，共 ${points.length} 轮，最优 ${formatScore(overallBest)}`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="iteration"
              tick={{ fontSize: 11 }}
              stroke="var(--muted-foreground)"
              label={{ value: "轮次", position: "insideBottomRight", offset: -2, fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 11 }}
              stroke="var(--muted-foreground)"
              tickFormatter={(value: number) => value.toFixed(1)}
            />
            <Tooltip
              formatter={(value, name) => [
                typeof value === "number" ? value.toFixed(3) : String(value),
                name === "score" ? "本轮分数" : "历史最优",
              ]}
              labelFormatter={(label) => `第 ${label} 轮`}
            />
            <Line
              type="monotone"
              dataKey="score"
              stroke="var(--chart-2)"
              strokeWidth={2}
              dot={{ r: 2.5 }}
              isAnimationActive={false}
            />
            <Line
              type="stepAfter"
              dataKey="best"
              stroke="var(--chart-3)"
              strokeWidth={1.5}
              strokeDasharray="5 3"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {bestRecord && (
        <section>
          <h3 className="mb-2 text-sm font-medium">
            最优一轮（第 {bestRecord.iteration + 1} 轮 · {formatScore(bestRecord.run_score.composite_mean)}）
          </h3>
          <table className="w-full text-left text-xs">
            <thead className="text-muted-foreground">
              <tr>
                <th scope="col" className="py-1 pr-3 font-normal">图片</th>
                <th scope="col" className="py-1 pr-3 font-normal">IoU</th>
                <th scope="col" className="py-1 pr-3 font-normal">误检</th>
                <th scope="col" className="py-1 pr-3 font-normal">漏检</th>
                <th scope="col" className="py-1 pr-3 font-normal">参考数</th>
                <th scope="col" className="py-1 font-normal">综合</th>
              </tr>
            </thead>
            <tbody>
              {bestRecord.run_score.image_scores.map((score) => (
                <tr key={score.image_id} className="border-t">
                  <td className="py-1 pr-3 font-mono">{score.image_id}</td>
                  <td className="py-1 pr-3 font-mono">{score.iou_mean.toFixed(3)}</td>
                  <td className="py-1 pr-3 font-mono">{score.false_positive_count}</td>
                  <td className="py-1 pr-3 font-mono">{score.false_negative_count}</td>
                  <td className="py-1 pr-3 font-mono">{score.ref_count}</td>
                  <td className="py-1 font-mono">{score.composite.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
