import { NextResponse } from "next/server";
import { jobStore } from "@/lib/store";

export const runtime = "nodejs";

/** GET /api/v1/health — Health check */
export async function GET() {
  return NextResponse.json({
    status: "healthy",
    version: "0.1.0",
    mode: "vercel-demo",
    jobs_in_memory: jobStore.size,
  });
}
