import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TrainingEvent } from "../types";

interface LossChartProps {
  data: TrainingEvent[];
}

// Recharts line chart displaying content / style / total losses on a log Y
// axis. Batch is a monotonically-increasing X axis — TrainingEvent.batch is
// global within an epoch, so we combine with epoch to get a stable number.
export function LossChart({ data }: LossChartProps) {
  // Keep only events that actually carry loss values. Terminal events like
  // ``{status: "done"}`` omit the loss fields and would break the log-scale
  // axis if we fed them in as undefined.
  const rows = useMemo(
    () =>
      data
        .filter((d) => typeof d.total_loss === "number")
        .map((d) => ({
          batch: d.batch,
          content_loss: d.content_loss,
          style_loss: d.style_loss,
          total_loss: d.total_loss,
        })),
    [data]
  );

  return (
    <div className="h-80 w-full rounded-lg border border-zinc-800 bg-zinc-900/50 p-3">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={rows}
          margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
        >
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
          <XAxis
            dataKey="batch"
            stroke="#a1a1aa"
            fontSize={11}
            tick={{ fill: "#a1a1aa" }}
          />
          <YAxis
            scale="log"
            domain={["auto", "auto"]}
            allowDataOverflow
            stroke="#a1a1aa"
            fontSize={11}
            tick={{ fill: "#a1a1aa" }}
          />
          <Tooltip
            contentStyle={{
              background: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: 6,
              fontSize: 12,
            }}
            labelStyle={{ color: "#e4e4e7" }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line
            type="monotone"
            dataKey="content_loss"
            stroke="#60a5fa"
            dot={false}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="style_loss"
            stroke="#f472b6"
            dot={false}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="total_loss"
            stroke="#34d399"
            dot={false}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default LossChart;
