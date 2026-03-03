import { NextRequest, NextResponse } from "next/server";
import { jobStore } from "@/lib/store";

export const runtime = "nodejs";

/** GET /api/v1/jobs/[id] — Get job details */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: jobId } = await params;
  const job = jobStore.get(jobId);

  if (!job) {
    return NextResponse.json({ detail: "Job not found" }, { status: 404 });
  }

  return NextResponse.json({
    job_id: job.job_id,
    status: job.status,
    current_stage: job.current_stage,
    progress: job.progress,
    source_language: job.source_language,
    source_language_confidence: job.source_language_confidence,
    target_languages: job.target_languages,
    speakers_detected: job.speakers_detected,
    segments_count: job.segments_count,
    duration_seconds: job.duration_seconds,
    error_message: job.error_message,
    output_paths: job.output_paths,
    created_at: job.created_at,
    completed_at: job.completed_at,
    stages: job.stages,
  });
}

/** DELETE /api/v1/jobs/[id] — Delete a job */
export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: jobId } = await params;

  if (!jobStore.has(jobId)) {
    return NextResponse.json({ detail: "Job not found" }, { status: 404 });
  }

  jobStore.delete(jobId);
  return new NextResponse(null, { status: 204 });
}
