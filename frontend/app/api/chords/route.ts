import { NextRequest } from "next/server";

import { proxyBackendGet } from "../_lib/backend";

export async function GET(request: NextRequest) {
  return proxyBackendGet("/chords", request);
}
