"use client";

interface StageInfo {
  name: string;
  label: string;
  status: "done" | "active" | "pending";
}

interface Props {
  stages: StageInfo[];
  overallProgress: number;
}

export default function JobProgress({ stages, overallProgress }: Props) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <div className="space-y-3">
        {stages.map((stage, idx) => (
          <div key={stage.name} className="flex items-center gap-3">
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                stage.status === "done"
                  ? "bg-green-100 text-green-700"
                  : stage.status === "active"
                  ? "bg-blue-100 text-blue-700 animate-pulse"
                  : "bg-gray-100 text-gray-400"
              }`}
            >
              {stage.status === "done" ? "\u2713" : idx + 1}
            </div>
            <span
              className={
                stage.status === "done"
                  ? "text-green-700"
                  : stage.status === "active"
                  ? "text-blue-700 font-medium"
                  : "text-gray-400"
              }
            >
              {stage.label}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-4 w-full bg-gray-200 rounded-full h-2">
        <div
          className="bg-blue-500 h-2 rounded-full transition-all duration-500"
          style={{ width: `${overallProgress}%` }}
        />
      </div>
    </div>
  );
}
