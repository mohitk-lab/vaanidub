/**
 * VaaniDub API client.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

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

  const job: Job = await resp.json();

  // Cache job in sessionStorage so detail page works across serverless instances
  try {
    sessionStorage.setItem(`vaanidub_job_${job.job_id}`, JSON.stringify(job));
  } catch {
    // Ignore storage errors
  }

  return job;
}

export async function getJob(jobId: string): Promise<Job> {
  // Try server first
  try {
    const resp = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`);
    if (resp.ok) {
      const job: Job = await resp.json();
      // Update cache
      try {
        sessionStorage.setItem(`vaanidub_job_${jobId}`, JSON.stringify(job));
      } catch { /* ignore */ }
      return job;
    }
  } catch {
    // Server unavailable, fall through to cache
  }

  // Fall back to sessionStorage cache
  try {
    const cached = sessionStorage.getItem(`vaanidub_job_${jobId}`);
    if (cached) return JSON.parse(cached);
  } catch { /* ignore */ }

  throw new Error("Job not found");
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
