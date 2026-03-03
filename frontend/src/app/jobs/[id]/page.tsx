"use client";

import { useEffect, useState } from "react";
import { getJob, getDownloadUrl, type Job } from "@/lib/api";
import { useParams } from "next/navigation";

const STAGE_LABELS: Record<string, string> = {
  ingest: "Ingesting Media",
  separate: "Separating Vocals",
  diarize: "Identifying Speakers",
  transcribe: "Transcribing Speech",
  prosody: "Analyzing Emotions",
  translate: "Translating",
  synthesize: "Cloning Voices",
  mixdown: "Final Mix",
};

const STAGE_ORDER = [
  "ingest", "separate", "diarize", "transcribe",
  "prosody", "translate", "synthesize", "mixdown",
];

export default function JobPage() {
  const params = useParams();
  const jobId = params.id as string;
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const fetchJob = async () => {
      try {
        const data = await getJob(jobId);
        setJob(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load job");
      }
    };

    fetchJob();

    // Poll every 3 seconds if not completed
    const interval = setInterval(async () => {
      try {
        const data = await getJob(jobId);
        setJob(data);
        if (data.status === "completed" || data.status === "failed") {
          clearInterval(interval);
        }
      } catch {
        // Ignore polling errors
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [jobId]);

  if (error) {
    return <p className="text-red-600">{error}</p>;
  }

  if (!job) {
    return <p className="text-gray-500">Loading...</p>;
  }

  const currentStageIdx = job.current_stage
    ? STAGE_ORDER.indexOf(job.current_stage)
    : -1;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Job {job.job_id}</h2>
          <p className="text-sm text-gray-500">
            Status:{" "}
            <span
              className={
                job.status === "completed"
                  ? "text-green-600 font-medium"
                  : job.status === "failed"
                  ? "text-red-600 font-medium"
                  : "text-blue-600 font-medium"
              }
            >
              {job.status}
            </span>
          </p>
        </div>
        {job.source_language && (
          <p className="text-sm text-gray-500">
            Source: <span className="font-medium">{job.source_language}</span>
          </p>
        )}
      </div>

      {/* Progress Pipeline */}
      <section className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-800 mb-4">Pipeline Progress</h3>
        <div className="space-y-3">
          {STAGE_ORDER.map((stage, idx) => {
            let status: "done" | "active" | "pending" = "pending";
            if (idx < currentStageIdx) status = "done";
            else if (idx === currentStageIdx) status = "active";

            if (job.status === "completed") status = "done";
            if (job.status === "failed" && idx <= currentStageIdx) {
              status = idx < currentStageIdx ? "done" : "active";
            }

            return (
              <div key={stage} className="flex items-center gap-3">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                    status === "done"
                      ? "bg-green-100 text-green-700"
                      : status === "active"
                      ? "bg-blue-100 text-blue-700 animate-pulse"
                      : "bg-gray-100 text-gray-400"
                  }`}
                >
                  {status === "done" ? "\u2713" : idx + 1}
                </div>
                <span
                  className={
                    status === "done"
                      ? "text-green-700"
                      : status === "active"
                      ? "text-blue-700 font-medium"
                      : "text-gray-400"
                  }
                >
                  {STAGE_LABELS[stage] || stage}
                </span>
              </div>
            );
          })}
        </div>

        {/* Overall progress bar */}
        <div className="mt-6">
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className={`h-2.5 rounded-full transition-all duration-500 ${
                job.status === "completed"
                  ? "bg-green-500"
                  : job.status === "failed"
                  ? "bg-red-500"
                  : "bg-blue-500"
              }`}
              style={{
                width: `${
                  job.status === "completed"
                    ? 100
                    : Math.max(5, ((currentStageIdx + 1) / 8) * 100)
                }%`,
              }}
            />
          </div>
        </div>
      </section>

      {/* Error Message */}
      {job.status === "failed" && job.error_message && (
        <section className="bg-red-50 border border-red-200 rounded-xl p-4">
          <p className="text-red-800 font-medium">Error</p>
          <p className="text-red-600 text-sm mt-1">{job.error_message}</p>
        </section>
      )}

      {/* Download Outputs */}
      {job.status === "completed" && job.output_paths && (
        <section className="bg-green-50 border border-green-200 rounded-xl p-6">
          <h3 className="font-semibold text-green-800 mb-4">Download Dubbed Output</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {Object.entries(job.output_paths).map(([lang]) => (
              <a
                key={lang}
                href={getDownloadUrl(job.job_id, lang)}
                download={`dubbed_${lang}.wav`}
                className="block p-3 bg-white rounded-lg border border-green-300 hover:bg-green-100 transition-colors text-center"
              >
                <p className="font-medium text-green-800">{lang.toUpperCase()}</p>
                <p className="text-sm text-green-600">Download</p>
              </a>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
