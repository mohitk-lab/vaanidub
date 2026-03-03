import { NextRequest, NextResponse } from "next/server";
import { jobStore, runDemoPipeline, type JobRecord } from "@/lib/store";

export const runtime = "nodejs";
export const maxDuration = 30; // seconds — allow time for WAV processing

/** POST /api/v1/jobs — Create a new dubbing job */
export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("file") as File | null;
  const targetLangsRaw = formData.get("target_languages") as string | null;
  const sourceLang = formData.get("source_language") as string | null;

  if (!file) {
    return NextResponse.json({ detail: "No file uploaded" }, { status: 400 });
  }
  if (!targetLangsRaw) {
    return NextResponse.json(
      { detail: "target_languages is required" },
      { status: 400 }
    );
  }

  let targets: string[];
  try {
    targets = JSON.parse(targetLangsRaw);
  } catch {
    return NextResponse.json(
      { detail: "target_languages must be a valid JSON array" },
      { status: 400 }
    );
  }

  if (!Array.isArray(targets) || targets.length === 0) {
    return NextResponse.json(
      { detail: "target_languages must be a non-empty array" },
      { status: 400 }
    );
  }

  const VALID_LANGS = [
    "hi", "ta", "te", "bn", "mr", "kn", "ml", "gu", "as", "or", "pa",
  ];
  for (const lang of targets) {
    if (!VALID_LANGS.includes(lang)) {
      return NextResponse.json(
        { detail: `Unsupported language: ${lang}` },
        { status: 400 }
      );
    }
  }

  // Read the uploaded file to base64
  const arrayBuffer = await file.arrayBuffer();
  const base64 = Buffer.from(arrayBuffer).toString("base64");

  // Generate short job ID
  const jobId = Math.random().toString(36).substring(2, 10);

  const job: JobRecord = {
    job_id: jobId,
    status: "pending",
    current_stage: null,
    progress: 0,
    source_language: sourceLang,
    source_language_confidence: null,
    target_languages: targets,
    speakers_detected: null,
    segments_count: null,
    duration_seconds: null,
    error_message: null,
    output_paths: null,
    stages: [],
    created_at: new Date().toISOString(),
    completed_at: null,
    input_base64: base64,
    input_filename: file.name,
  };

  jobStore.set(jobId, job);

  // Run pipeline synchronously (serverless can't do background work)
  runDemoPipeline(jobId);

  // Return the completed job
  return NextResponse.json(
    {
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
      output_buffers: job.output_buffers,
      stages: job.stages,
      created_at: job.created_at,
      completed_at: job.completed_at,
    },
    { status: 201 }
  );
}

/** GET /api/v1/jobs — List jobs */
export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const page = parseInt(searchParams.get("page") || "1");
  const perPage = parseInt(searchParams.get("per_page") || "20");
  const statusFilter = searchParams.get("status");

  let jobs = Array.from(jobStore.values());

  if (statusFilter) {
    jobs = jobs.filter((j) => j.status === statusFilter);
  }

  // Sort by created_at descending
  jobs.sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );

  const total = jobs.length;
  const start = (page - 1) * perPage;
  const paged = jobs.slice(start, start + perPage);

  return NextResponse.json({
    jobs: paged.map((j) => ({
      job_id: j.job_id,
      status: j.status,
      current_stage: j.current_stage,
      progress: j.progress,
      target_languages: j.target_languages,
      source_language: j.source_language,
      duration_seconds: j.duration_seconds,
      created_at: j.created_at,
      completed_at: j.completed_at,
    })),
    total,
    page,
    per_page: perPage,
  });
}
