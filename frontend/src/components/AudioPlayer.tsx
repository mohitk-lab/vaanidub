"use client";

interface Props {
  label: string;
  src: string;
}

export default function AudioPlayer({ label, src }: Props) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <p className="text-sm font-medium text-gray-700 mb-2">{label}</p>
      <audio controls className="w-full" src={src}>
        Your browser does not support audio playback.
      </audio>
    </div>
  );
}
