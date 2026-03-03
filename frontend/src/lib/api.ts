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
  /** Base64-encoded WAV outputs keyed by language code */
  output_buffers?: Record<string, string>;
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
    let message = "Failed to create job";
    try {
      const text = await resp.text();
      try {
        const err = JSON.parse(text);
        message = err.detail || message;
      } catch {
        // Plain text error (e.g. Vercel "Request Entity Too Large")
        if (text.includes("Request Entity Too Large") || resp.status === 413) {
          message = "File too large for serverless. Try a smaller file (under 4MB).";
        } else {
          message = text || `Server error (${resp.status})`;
        }
      }
    } catch {
      message = `Server error (${resp.status})`;
    }
    throw new Error(message);
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
      // Merge with cached output_buffers (server may not have them on a different instance)
      try {
        const cached = sessionStorage.getItem(`vaanidub_job_${jobId}`);
        if (cached) {
          const cachedJob: Job = JSON.parse(cached);
          if (cachedJob.output_buffers && !job.output_buffers) {
            job.output_buffers = cachedJob.output_buffers;
          }
        }
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

/**
 * Get download URL for a dubbed output.
 * Prefers client-side Blob URL from cached data, falls back to server URL.
 */
export function getDownloadUrl(jobId: string, langCode: string): string {
  // Try to create a Blob URL from cached output_buffers
  try {
    const cached = sessionStorage.getItem(`vaanidub_job_${jobId}`);
    if (cached) {
      const job: Job = JSON.parse(cached);
      const base64 = job.output_buffers?.[langCode];
      if (base64) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: "audio/wav" });
        return URL.createObjectURL(blob);
      }
    }
  } catch { /* ignore */ }

  // Fall back to server URL
  return `${API_BASE}/api/v1/jobs/${jobId}/output/${langCode}`;
}
