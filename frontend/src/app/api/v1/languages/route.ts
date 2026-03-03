import { NextResponse } from "next/server";

export const runtime = "nodejs";

const LANGUAGES = [
  { code: "hi", name: "Hindi", native_name: "हिन्दी", script: "Devanagari", tts_providers: ["indicf5", "elevenlabs"] },
  { code: "ta", name: "Tamil", native_name: "தமிழ்", script: "Tamil", tts_providers: ["indicf5", "elevenlabs"] },
  { code: "te", name: "Telugu", native_name: "తెలుగు", script: "Telugu", tts_providers: ["indicf5", "elevenlabs"] },
  { code: "bn", name: "Bengali", native_name: "বাংলা", script: "Bengali", tts_providers: ["indicf5", "elevenlabs"] },
  { code: "mr", name: "Marathi", native_name: "मराठी", script: "Devanagari", tts_providers: ["indicf5", "elevenlabs"] },
  { code: "kn", name: "Kannada", native_name: "ಕನ್ನಡ", script: "Kannada", tts_providers: ["indicf5", "elevenlabs"] },
  { code: "ml", name: "Malayalam", native_name: "മലയാളം", script: "Malayalam", tts_providers: ["indicf5", "elevenlabs"] },
  { code: "gu", name: "Gujarati", native_name: "ગુજરાતી", script: "Gujarati", tts_providers: ["indicf5", "elevenlabs"] },
  { code: "as", name: "Assamese", native_name: "অসমীয়া", script: "Bengali", tts_providers: ["indicf5"] },
  { code: "or", name: "Odia", native_name: "ଓଡ଼ିଆ", script: "Odia", tts_providers: ["indicf5"] },
  { code: "pa", name: "Punjabi", native_name: "ਪੰਜਾਬੀ", script: "Gurmukhi", tts_providers: ["indicf5"] },
];

/** GET /api/v1/languages — List supported languages */
export async function GET() {
  return NextResponse.json({ languages: LANGUAGES });
}
