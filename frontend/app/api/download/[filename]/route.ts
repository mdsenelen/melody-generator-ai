import { NextRequest } from "next/server";

import { proxyBackendGet } from "../../_lib/backend";

type DownloadRouteContext = {
  params: Promise<{ filename: string }>;
};

export async function GET(
  request: NextRequest,
  context: DownloadRouteContext,
) {
  const { filename } = await context.params;
  return proxyBackendGet(`/download/${encodeURIComponent(filename)}`, request);
}
