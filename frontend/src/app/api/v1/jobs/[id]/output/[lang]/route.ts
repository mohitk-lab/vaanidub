import { NextRequest, NextResponse } from "next/server";
import { jobStore } from "@/lib/store";

export const runtime = "nodejs";

/** GET /api/v1/jobs/[id]/output/[lang] — Download dubbed output */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string; lang: string }> }
) {
  const { id: jobId, lang } = await params;
  const job = jobStore.get(jobId);

  if (!job) {
    return NextResponse.json({ detail: "Job not found" }, { status: 404 });
  }

  if (job.status !== "completed") {
    return NextResponse.json(
      { detail: `Job is not completed (status: ${job.status})` },
      { status: 400 }
    );
  }

  const base64 = job.output_buffers?.[lang];
  if (!base64) {
    return NextResponse.json(
      { detail: `Output not found for language: ${lang}` },
      { status: 404 }
    );
  }

  const buffer = Buffer.from(base64, "base64");

  return new NextResponse(buffer, {
    status: 200,
    headers: {
      "Content-Type": "audio/wav",
      "Content-Disposition": `attachment; filename="dubbed_${lang}.wav"`,
      "Content-Length": buffer.length.toString(),
    },
  });
}
