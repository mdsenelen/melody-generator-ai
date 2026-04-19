import { NextRequest } from "next/server";

import { proxyBackendRequest } from "../_lib/backend";

export async function POST(request: NextRequest) {
  return proxyBackendRequest(request, "/generate-progression");
}
