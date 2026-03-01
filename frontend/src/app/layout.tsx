import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "VaaniDub — AI Regional Dubbing",
  description: "Professional AI dubbing for Indian languages with voice cloning",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 min-h-screen">
        <header className="bg-white border-b border-gray-200">
          <div className="max-w-5xl mx-auto px-4 py-4">
            <h1 className="text-2xl font-bold text-gray-900">
              VaaniDub
              <span className="text-sm font-normal text-gray-500 ml-2">
                AI Regional Dubbing
              </span>
            </h1>
          </div>
        </header>
        <main className="max-w-5xl mx-auto px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
