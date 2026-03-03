/**
 * Client-side audio trimmer for large file uploads.
 *
 * For WAV files: efficient byte-level slicing (no full decode needed).
 * For compressed formats: AudioContext decode on a sliced chunk.
 *
 * This lets users "upload" 1GB+ files — the browser trims to the first
 * N seconds and sends a compact 16kHz mono WAV under the 4.5MB limit.
 */

const TARGET_SAMPLE_RATE = 16000;
const TARGET_DURATION_SEC = 60; // max seconds to keep
const MAX_UPLOAD_BYTES = 4 * 1024 * 1024; // 4MB final upload limit

/** Small enough to upload directly without trimming */
const DIRECT_UPLOAD_LIMIT = 3.5 * 1024 * 1024; // 3.5MB

/**
 * Prepare a file for upload. Returns the original file if small enough,
 * or a trimmed WAV if the file is too large.
 */
export async function prepareFileForUpload(
  file: File,
  onProgress?: (msg: string) => void
): Promise<File> {
  // Small files: upload directly
  if (file.size <= DIRECT_UPLOAD_LIMIT) {
    return file;
  }

  const ext = file.name.split(".").pop()?.toLowerCase() || "";

  onProgress?.("Trimming audio for upload...");

  if (ext === "wav") {
    return trimWavFile(file, onProgress);
  } else {
    return trimWithAudioContext(file, onProgress);
  }
}

// ── WAV byte-level slicing (very memory efficient) ──

async function trimWavFile(
  file: File,
  onProgress?: (msg: string) => void
): Promise<File> {
  onProgress?.("Reading WAV header...");

  // Read header (first 44 bytes)
  const headerBuf = await file.slice(0, 44).arrayBuffer();
  const headerView = new DataView(headerBuf);

  // Parse WAV header
  const numChannels = headerView.getUint16(22, true);
  const sampleRate = headerView.getUint32(24, true);
  const bitsPerSample = headerView.getUint16(34, true);
  const bytesPerSample = (bitsPerSample / 8) * numChannels;
  const bytesPerSecond = sampleRate * bytesPerSample;

  // Calculate how many bytes for TARGET_DURATION_SEC
  const maxDataBytes = Math.min(
    TARGET_DURATION_SEC * bytesPerSecond,
    file.size - 44
  );

  // Also ensure final WAV fits in upload limit
  const maxForLimit = MAX_UPLOAD_BYTES - 44;
  const dataBytes = Math.min(maxDataBytes, maxForLimit);

  onProgress?.(
    `Trimming to ${Math.round(dataBytes / bytesPerSecond)}s of ${Math.round((file.size - 44) / bytesPerSecond)}s...`
  );

  // Read only the bytes we need
  const dataBuf = await file.slice(44, 44 + dataBytes).arrayBuffer();

  // If sample rate is already 16kHz mono, just re-header
  if (sampleRate === TARGET_SAMPLE_RATE && numChannels === 1) {
    const wav = buildWav(new Float32Array(0), TARGET_SAMPLE_RATE, dataBuf);
    return new File([wav], file.name.replace(/\.wav$/i, "_trimmed.wav"), {
      type: "audio/wav",
    });
  }

  // Otherwise, decode the trimmed chunk and resample
  onProgress?.("Resampling to 16kHz mono...");
  const fullBuf = concatBuffers(headerBuf, dataBuf);
  const audioCtx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
  try {
    const decoded = await audioCtx.decodeAudioData(fullBuf);
    const mono = decoded.getChannelData(0);
    const wav = buildWav(mono, TARGET_SAMPLE_RATE);
    return new File([wav], file.name.replace(/\.wav$/i, "_trimmed.wav"), {
      type: "audio/wav",
    });
  } finally {
    await audioCtx.close();
  }
}

// ── Compressed format trimming (MP3, FLAC, etc.) ──

async function trimWithAudioContext(
  file: File,
  onProgress?: (msg: string) => void
): Promise<File> {
  // For compressed formats, read enough to get ~60s of audio.
  // MP3 at 128kbps = ~960KB/min, so 10MB should cover 10+ minutes.
  // For higher bitrates or FLAC, we read more.
  const readSize = Math.min(file.size, 20 * 1024 * 1024); // max 20MB
  onProgress?.(
    `Reading ${(readSize / (1024 * 1024)).toFixed(1)}MB of ${(file.size / (1024 * 1024)).toFixed(1)}MB...`
  );

  const chunk = await file.slice(0, readSize).arrayBuffer();

  onProgress?.("Decoding audio...");
  const audioCtx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
  try {
    const decoded = await audioCtx.decodeAudioData(chunk);
    const maxSamples = TARGET_DURATION_SEC * TARGET_SAMPLE_RATE;
    const channelData = decoded.getChannelData(0);
    const mono = channelData.length > maxSamples
      ? channelData.slice(0, maxSamples)
      : channelData;

    const duration = mono.length / TARGET_SAMPLE_RATE;
    onProgress?.(`Encoding ${Math.round(duration)}s as WAV...`);

    const wav = buildWav(mono, TARGET_SAMPLE_RATE);
    const baseName = file.name.replace(/\.[^.]+$/, "");
    return new File([wav], `${baseName}_trimmed.wav`, { type: "audio/wav" });
  } finally {
    await audioCtx.close();
  }
}

// ── WAV encoding helpers ──

function buildWav(
  samples: Float32Array,
  sampleRate: number,
  rawPcmData?: ArrayBuffer
): ArrayBuffer {
  const useRaw = rawPcmData && rawPcmData.byteLength > 0;
  const dataSize = useRaw ? rawPcmData.byteLength : samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");

  // fmt chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample

  // data chunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  if (useRaw) {
    new Uint8Array(buffer, 44).set(new Uint8Array(rawPcmData));
  } else {
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, Math.round(s * 32767), true);
    }
  }

  return buffer;
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

function concatBuffers(a: ArrayBuffer, b: ArrayBuffer): ArrayBuffer {
  const result = new ArrayBuffer(a.byteLength + b.byteLength);
  const view = new Uint8Array(result);
  view.set(new Uint8Array(a), 0);
  view.set(new Uint8Array(b), a.byteLength);
  return result;
}
