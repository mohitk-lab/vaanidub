/**
 * In-memory job store for Vercel serverless deployment.
 *
 * Jobs persist as long as the serverless function instance stays warm.
 * For a production deployment you'd swap this for a real database.
 */

export interface StageLog {
  name: string;
  status: string;
  duration_sec: number | null;
  provider_used: string;
}

export interface JobRecord {
  job_id: string;
  status: string;
  current_stage: string | null;
  progress: number;
  source_language: string | null;
  source_language_confidence: number | null;
  target_languages: string[];
  speakers_detected: number | null;
  segments_count: number | null;
  duration_seconds: number | null;
  error_message: string | null;
  output_paths: Record<string, string> | null;
  stages: StageLog[];
  created_at: string;
  completed_at: string | null;
  /** Base64-encoded uploaded audio for later download */
  input_base64?: string;
  input_filename?: string;
  /** Base64-encoded output WAVs keyed by language */
  output_buffers?: Record<string, string>;
}

// Global in-memory store — survives across requests while the function is warm
const globalStore = globalThis as typeof globalThis & {
  __vaanidub_jobs?: Map<string, JobRecord>;
};

if (!globalStore.__vaanidub_jobs) {
  globalStore.__vaanidub_jobs = new Map();
}

export const jobStore = globalStore.__vaanidub_jobs;

// ── Demo pipeline constants ──

const STAGE_NAMES = [
  "ingest",
  "separate",
  "diarize",
  "transcribe",
  "prosody",
  "translate",
  "synthesize",
  "mixdown",
] as const;

const STAGE_PROGRESS: Record<string, number> = {
  ingest: 0.05,
  separate: 0.15,
  diarize: 0.25,
  transcribe: 0.40,
  prosody: 0.50,
  translate: 0.60,
  synthesize: 0.75,
  mixdown: 0.90,
};

const DEMO_TRANSLATIONS: Record<string, string> = {
  hi: "यह एक डेमो डबिंग है। वाणीडब आपकी आवाज़ को हिंदी में बदल रहा है।",
  ta: "இது ஒரு டெமோ டப்பிங். வாணிடப் உங்கள் குரலை தமிழில் மாற்றுகிறது.",
  te: "ఇది ఒక డెమో డబ్బింగ్. వాణీడబ్ మీ గొంతును తెలుగులోకి మారుస్తోంది.",
  bn: "এটি একটি ডেমো ডাবিং। ভানিডাব আপনার কণ্ঠকে বাংলায় রূপান্তর করছে।",
  mr: "हा एक डेमो डबिंग आहे. वाणीडब तुमचा आवाज मराठीत बदलत आहे.",
  kn: "ಇದು ಡೆಮೊ ಡಬ್ಬಿಂಗ್. ವಾಣಿಡಬ್ ನಿಮ್ಮ ಧ್ವನಿಯನ್ನು ಕನ್ನಡಕ್ಕೆ ಬದಲಾಯಿಸುತ್ತಿದೆ.",
  ml: "ഇത് ഒരു ഡെമോ ഡബ്ബിംഗ് ആണ്. വാണിഡബ് നിങ്ങളുടെ ശബ്ദം മലയാളത്തിലേക്ക് മാറ്റുന്നു.",
  gu: "આ એક ડેમો ડબિંગ છે. વાણીડબ તમારો અવાજ ગુજરાતીમાં બદલી રહ્યું છે.",
  as: "এইটো এটা ডেমো ডাবিং। ভানিডাব আপোনাৰ কণ্ঠক অসমীয়ালৈ সলনি কৰিছে।",
  or: "ଏହା ଏକ ଡେମୋ ଡବିଂ। ଭାନୀଡବ ଆପଣଙ୍କ କଣ୍ଠକୁ ଓଡ଼ିଆକୁ ବଦଳାଉଛି।",
  pa: "ਇਹ ਇੱਕ ਡੈਮੋ ਡਬਿੰਗ ਹੈ। ਵਾਣੀਡੱਬ ਤੁਹਾਡੀ ਆਵਾਜ਼ ਨੂੰ ਪੰਜਾਬੀ ਵਿੱਚ ਬਦਲ ਰਿਹਾ ਹੈ।",
};

// Pitch shift factors per language (used for WAV generation)
const PITCH_FACTORS: Record<string, number> = {
  hi: 1.08, ta: 0.94, te: 1.12, bn: 0.92, mr: 1.06,
  kn: 0.88, ml: 1.15, gu: 0.97, as: 1.10, or: 0.93, pa: 1.05,
};

// ── WAV helpers (pure JS, no native deps) ──

function createWavBuffer(samples: Float32Array, sampleRate: number): Buffer {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const headerSize = 44;
  const buffer = Buffer.alloc(headerSize + dataSize);

  // RIFF header
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);

  // fmt chunk
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitsPerSample, 34);

  // data chunk
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);

  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    buffer.writeInt16LE(Math.round(s * 32767), headerSize + i * 2);
  }
  return buffer;
}

function parseWavSamples(base64: string): { samples: Float32Array; sampleRate: number } {
  const buf = Buffer.from(base64, "base64");

  // Read header
  const numChannels = buf.readUInt16LE(22);
  const sampleRate = buf.readUInt32LE(24);
  const bitsPerSample = buf.readUInt16LE(34);
  const dataOffset = 44;
  const bytesPerSample = bitsPerSample / 8;
  const numSamples = (buf.length - dataOffset) / (bytesPerSample * numChannels);
  const samples = new Float32Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    const offset = dataOffset + i * bytesPerSample * numChannels;
    if (bitsPerSample === 16) {
      samples[i] = buf.readInt16LE(offset) / 32768;
    } else if (bitsPerSample === 32) {
      samples[i] = buf.readFloatLE(offset);
    } else {
      // 8-bit unsigned
      samples[i] = (buf.readUInt8(offset) - 128) / 128;
    }
  }

  return { samples, sampleRate };
}

function pitchShift(samples: Float32Array, factor: number): Float32Array {
  if (Math.abs(factor - 1.0) < 0.01) return samples;
  const newLen = Math.round(samples.length / factor);
  const result = new Float32Array(samples.length);
  // Simple linear interpolation resample
  for (let i = 0; i < newLen; i++) {
    const srcIdx = (i * factor);
    const idx0 = Math.floor(srcIdx);
    const idx1 = Math.min(idx0 + 1, samples.length - 1);
    const frac = srcIdx - idx0;
    result[i] = samples[idx0] * (1 - frac) + samples[idx1] * frac;
  }
  // Resample back to original length
  const final = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const srcIdx = (i / samples.length) * newLen;
    const idx0 = Math.floor(srcIdx);
    const idx1 = Math.min(idx0 + 1, newLen - 1);
    const frac = srcIdx - idx0;
    final[i] = result[idx0] * (1 - frac) + result[idx1] * frac;
  }
  return final;
}

function generateDemoWav(durationSec: number, sampleRate: number, pitchFactor: number): Buffer {
  const numSamples = Math.round(durationSec * sampleRate);
  const samples = new Float32Array(numSamples);

  // Generate a pleasant chord (major triad) with fade in/out
  const f1 = 261.63 * pitchFactor; // C4
  const f2 = 329.63 * pitchFactor; // E4
  const f3 = 392.00 * pitchFactor; // G4

  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;
    samples[i] =
      0.2 * Math.sin(2 * Math.PI * f1 * t) +
      0.15 * Math.sin(2 * Math.PI * f2 * t) +
      0.1 * Math.sin(2 * Math.PI * f3 * t);
  }

  // Fade in/out
  const fadeSamples = Math.min(Math.round(0.3 * sampleRate), numSamples / 2);
  for (let i = 0; i < fadeSamples; i++) {
    const gain = i / fadeSamples;
    samples[i] *= gain;
    samples[numSamples - 1 - i] *= gain;
  }

  return createWavBuffer(samples, sampleRate);
}

// ── Demo pipeline (runs async, updates job store in stages) ──

export async function runDemoPipeline(jobId: string): Promise<void> {
  const job = jobStore.get(jobId);
  if (!job) return;

  job.status = "processing";

  const SAMPLE_RATE = 16000;
  const durationSec = 10; // default demo duration

  // Determine actual duration from uploaded WAV if available
  let actualDuration = durationSec;
  let inputSamples: Float32Array | null = null;
  let inputSampleRate = SAMPLE_RATE;

  if (job.input_base64) {
    try {
      const parsed = parseWavSamples(job.input_base64);
      inputSamples = parsed.samples;
      inputSampleRate = parsed.sampleRate;
      actualDuration = inputSamples.length / inputSampleRate;
    } catch {
      // If parsing fails, use generated audio
    }
  }

  try {
    for (let i = 0; i < STAGE_NAMES.length; i++) {
      const stage = STAGE_NAMES[i];
      job.current_stage = stage;
      job.progress = STAGE_PROGRESS[stage];
      job.stages.push({
        name: stage,
        status: "completed",
        duration_sec: [1, 1, 0.5, 0.5, 0.3, 0.5, 1, 1][i],
        provider_used: "demo",
      });

      // Small delay to simulate processing (visible during polling)
      await new Promise((r) => setTimeout(r, 400));
    }

    // Build per-language output WAVs
    const outputPaths: Record<string, string> = {};
    const outputBuffers: Record<string, string> = {};

    for (const lang of job.target_languages) {
      const factor = PITCH_FACTORS[lang] || 1.0;

      let wavBuf: Buffer;
      if (inputSamples) {
        // Pitch-shift the actual uploaded audio
        const shifted = pitchShift(inputSamples, factor);
        wavBuf = createWavBuffer(shifted, inputSampleRate);
      } else {
        wavBuf = generateDemoWav(actualDuration, SAMPLE_RATE, factor);
      }

      outputBuffers[lang] = wavBuf.toString("base64");
      outputPaths[lang] = `dubbed_${lang}.wav`;
    }

    job.status = "completed";
    job.current_stage = "completed";
    job.progress = 1.0;
    job.source_language = "en";
    job.source_language_confidence = 0.95;
    job.duration_seconds = actualDuration;
    job.speakers_detected = 1;
    job.segments_count = Math.ceil(actualDuration / 5);
    job.output_paths = outputPaths;
    job.output_buffers = outputBuffers;
    job.completed_at = new Date().toISOString();
  } catch (e) {
    job.status = "failed";
    job.error_message = e instanceof Error ? e.message : "Pipeline failed";
  }
}
