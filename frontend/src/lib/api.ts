/**
 * VaaniDub API client.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Job {
  job_id: string;
  status: string;
  current_stage: string | null;
  progress: number;
  source_language: string | null;
  target_languages: string[];
  duration_seconds: number | null;
  error_message: string | null;
  output_paths: Record<string, string> | null;
  created_at: string | null;
  completed_at: string | null;
  stages?: StageStatus[];
}

export interface StageStatus {
  name: string;
  status: string;
  duration_sec: number | null;
}

export interface Language {
  code: string;
  name: string;
  native_name: string;
  script: string;
  tts_providers: string[];
}

export async function createJob(
  file: File,
  targetLanguages: string[],
  sourceLanguage?: string
): Promise<Job> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("target_languages", JSON.stringify(targetLanguages));
  if (sourceLanguage) {
    formData.append("source_language", sourceLanguage);
  }

  const resp = await fetch(`${API_BASE}/api/v1/jobs`, {
    method: "POST",
    body: formData,
  });

  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail || "Failed to create job");
  }

  return resp.json();
}

export async function getJob(jobId: string): Promise<Job> {
  const resp = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`);
  if (!resp.ok) throw new Error("Job not found");
  return resp.json();
}

export async function listJobs(page = 1): Promise<{ jobs: Job[]; total: number }> {
  const resp = await fetch(`${API_BASE}/api/v1/jobs?page=${page}`);
  if (!resp.ok) throw new Error("Failed to list jobs");
  return resp.json();
}

export async function getLanguages(): Promise<Language[]> {
  const resp = await fetch(`${API_BASE}/api/v1/languages`);
  if (!resp.ok) throw new Error("Failed to fetch languages");
  const data = await resp.json();
  return data.languages;
}

export function getDownloadUrl(jobId: string, langCode: string): string {
  return `${API_BASE}/api/v1/jobs/${jobId}/output/${langCode}`;
}
