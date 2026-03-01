"use client";

interface Language {
  code: string;
  name: string;
  native_name: string;
}

const LANGUAGES: Language[] = [
  { code: "hi", name: "Hindi", native_name: "हिन्दी" },
  { code: "ta", name: "Tamil", native_name: "தமிழ்" },
  { code: "te", name: "Telugu", native_name: "తెలుగు" },
  { code: "bn", name: "Bengali", native_name: "বাংলা" },
  { code: "mr", name: "Marathi", native_name: "मराठी" },
  { code: "kn", name: "Kannada", native_name: "ಕನ್ನಡ" },
  { code: "ml", name: "Malayalam", native_name: "മലയാളം" },
  { code: "gu", name: "Gujarati", native_name: "ગુજરાતી" },
  { code: "as", name: "Assamese", native_name: "অসমীয়া" },
  { code: "or", name: "Odia", native_name: "ଓଡ଼ିଆ" },
  { code: "pa", name: "Punjabi", native_name: "ਪੰਜਾਬੀ" },
];

interface Props {
  selected: string[];
  onChange: (selected: string[]) => void;
}

export default function LanguageSelector({ selected, onChange }: Props) {
  const toggle = (code: string) => {
    onChange(
      selected.includes(code)
        ? selected.filter((c) => c !== code)
        : [...selected, code]
    );
  };

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
      {LANGUAGES.map((lang) => (
        <button
          key={lang.code}
          onClick={() => toggle(lang.code)}
          className={`p-3 rounded-lg border text-left transition-all ${
            selected.includes(lang.code)
              ? "border-blue-500 bg-blue-50 ring-2 ring-blue-200"
              : "border-gray-200 hover:border-gray-300"
          }`}
        >
          <p className="font-medium text-gray-800">{lang.native_name}</p>
          <p className="text-sm text-gray-500">{lang.name}</p>
        </button>
      ))}
    </div>
  );
}
