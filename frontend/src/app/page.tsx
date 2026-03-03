"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { createJob, type Language } from "@/lib/api";
import { prepareFileForUpload } from "@/lib/audio-trimmer";
import { useRouter } from "next/navigation";

const LANGUAGES: Language[] = [
  { code: "hi", name: "Hindi", native_name: "हिन्दी", script: "Devanagari", tts_providers: [] },
  { code: "ta", name: "Tamil", native_name: "தமிழ்", script: "Tamil", tts_providers: [] },
  { code: "te", name: "Telugu", native_name: "తెలుగు", script: "Telugu", tts_providers: [] },
  { code: "bn", name: "Bengali", native_name: "বাংলা", script: "Bengali", tts_providers: [] },
  { code: "mr", name: "Marathi", native_name: "मराठी", script: "Devanagari", tts_providers: [] },
  { code: "kn", name: "Kannada", native_name: "ಕನ್ನಡ", script: "Kannada", tts_providers: [] },
  { code: "ml", name: "Malayalam", native_name: "മലയാളം", script: "Malayalam", tts_providers: [] },
  { code: "gu", name: "Gujarati", native_name: "ગુજરાતી", script: "Gujarati", tts_providers: [] },
  { code: "as", name: "Assamese", native_name: "অসমীয়া", script: "Bengali", tts_providers: [] },
  { code: "or", name: "Odia", native_name: "ଓଡ଼ିଆ", script: "Odia", tts_providers: [] },
  { code: "pa", name: "Punjabi", native_name: "ਪੰਜਾਬੀ", script: "Gurmukhi", tts_providers: [] },
];

const MAX_FILE_SIZE = 1024 * 1024 * 1024; // 1GB

export default function HomePage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [selectedLangs, setSelectedLangs] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const f = acceptedFiles[0];
      if (f.size > MAX_FILE_SIZE) {
        setError("File exceeds 1GB limit.");
        return;
      }
      setFile(f);
      setError(null);
      setStatusMsg(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "audio/*": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"],
      "video/*": [".mp4", ".mkv", ".avi", ".webm", ".mov"],
    },
    maxFiles: 1,
  });

  const toggleLang = (code: string) => {
    setSelectedLangs((prev) =>
      prev.includes(code) ? prev.filter((l) => l !== code) : [...prev, code]
    );
  };

  const handleSubmit = async () => {
    if (!file || selectedLangs.length === 0) {
      setError("Please select a file and at least one target language.");
      return;
    }

    setLoading(true);
    setError(null);
    setStatusMsg(null);

    try {
      // Trim large files client-side before upload
      const prepared = await prepareFileForUpload(file, (msg) =>
        setStatusMsg(msg)
      );

      if (prepared !== file) {
        setStatusMsg(
          `Trimmed: ${(file.size / (1024 * 1024)).toFixed(1)}MB → ${(prepared.size / (1024 * 1024)).toFixed(1)}MB`
        );
      } else {
        setStatusMsg("Uploading...");
      }

      const job = await createJob(prepared, selectedLangs);
      router.push(`/jobs/${job.job_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create job");
      setStatusMsg(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* File Upload */}
      <section>
        <h2 className="text-lg font-semibold text-gray-800 mb-3">1. Upload Audio or Video</h2>
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
            isDragActive
              ? "border-blue-500 bg-blue-50"
              : file
              ? "border-green-400 bg-green-50"
              : "border-gray-300 hover:border-gray-400"
          }`}
        >
          <input {...getInputProps()} />
          {file ? (
            <div>
              <p className="text-green-700 font-medium">{file.name}</p>
              <p className="text-sm text-gray-500 mt-1">
                {file.size >= 1024 * 1024
                  ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
                  : `${(file.size / 1024).toFixed(0)} KB`}
                {file.size > 3.5 * 1024 * 1024 && (
                  <span className="text-amber-600 ml-2">
                    (will be trimmed to first 60s for demo)
                  </span>
                )}
              </p>
              <p className="text-xs text-gray-400 mt-2">Click or drag to replace</p>
            </div>
          ) : (
            <div>
              <p className="text-gray-600">Drag & drop your audio/video file here</p>
              <p className="text-sm text-gray-400 mt-1">
                MP3, WAV, FLAC, MP4, MKV — up to 1GB
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Language Selection */}
      <section>
        <h2 className="text-lg font-semibold text-gray-800 mb-3">
          2. Select Target Languages
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {LANGUAGES.map((lang) => (
            <button
              key={lang.code}
              onClick={() => toggleLang(lang.code)}
              className={`p-3 rounded-lg border text-left transition-all ${
                selectedLangs.includes(lang.code)
                  ? "border-blue-500 bg-blue-50 ring-2 ring-blue-200"
                  : "border-gray-200 hover:border-gray-300"
              }`}
            >
              <p className="font-medium text-gray-800">{lang.native_name}</p>
              <p className="text-sm text-gray-500">{lang.name}</p>
            </button>
          ))}
        </div>
        {selectedLangs.length > 0 && (
          <p className="text-sm text-gray-500 mt-2">
            Selected: {selectedLangs.map((c) => LANGUAGES.find((l) => l.code === c)?.name).join(", ")}
          </p>
        )}
      </section>

      {/* Submit */}
      <section>
        {error && (
          <p className="text-red-600 text-sm mb-3">{error}</p>
        )}
        {statusMsg && loading && (
          <p className="text-blue-600 text-sm mb-3 animate-pulse">{statusMsg}</p>
        )}
        <button
          onClick={handleSubmit}
          disabled={loading || !file || selectedLangs.length === 0}
          className="w-full py-3 px-6 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Processing... please wait" : "Start Dubbing"}
        </button>
      </section>
    </div>
  );
}
