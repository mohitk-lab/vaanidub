"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
}

export default function FileUploader({ onFileSelect, selectedFile }: FileUploaderProps) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) onFileSelect(accepted[0]);
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "audio/*": [".mp3", ".wav", ".flac", ".aac"],
      "video/*": [".mp4", ".mkv", ".avi", ".webm"],
    },
    maxFiles: 1,
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
        isDragActive
          ? "border-blue-500 bg-blue-50"
          : selectedFile
          ? "border-green-400 bg-green-50"
          : "border-gray-300 hover:border-gray-400"
      }`}
    >
      <input {...getInputProps()} />
      {selectedFile ? (
        <div>
          <p className="text-green-700 font-medium">{selectedFile.name}</p>
          <p className="text-sm text-gray-500">
            {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
          </p>
        </div>
      ) : (
        <p className="text-gray-500">
          {isDragActive ? "Drop file here..." : "Drag & drop audio/video file"}
        </p>
      )}
    </div>
  );
}
