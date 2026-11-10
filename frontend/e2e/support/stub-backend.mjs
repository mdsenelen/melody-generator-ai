// The FastAPI stand-in for requests the browser can't intercept: a download
// is a navigation, so it skips page.route and reaches the Next.js
// /api/download proxy, which fetches from the backend server-side. Next's
// BACKEND_BASE_URL points here (playwright.config.ts). Serves fixture MIDI
// files and /health; everything else is a 501, never a real backend.
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixture = (name) => JSON.parse(readFileSync(path.join(here, "..", "fixtures", name), "utf8"));

const midiFiles = new Map();
for (const { midi_filename, midi_b64 } of fixture("variants.json").variants) {
  midiFiles.set(midi_filename, Buffer.from(midi_b64, "base64"));
}
const transcription = fixture("transcription.json");
midiFiles.set(transcription.midi_filename, Buffer.from(transcription.midi_b64, "base64"));

const port = Number(process.argv[2] ?? 8100);

createServer((request, response) => {
  const { pathname } = new URL(request.url, `http://127.0.0.1:${port}`);
  const send = (status, body, headers = { "content-type": "application/json" }) => {
    response.writeHead(status, headers);
    response.end(Buffer.isBuffer(body) ? body : JSON.stringify(body));
  };

  if (pathname === "/health") return send(200, { status: "ok" });

  const download = pathname.match(/^\/api\/download\/([^/]+)$/);
  if (request.method === "GET" && download) {
    const filename = decodeURIComponent(download[1]);
    const bytes = midiFiles.get(filename);
    if (!bytes) return send(404, { detail: "File not found" });
    return send(200, bytes, {
      "content-type": "audio/midi",
      "content-disposition": `attachment; filename="${filename}"`,
    });
  }

  send(501, { detail: `e2e stub backend: unmocked ${request.method} ${pathname}` });
}).listen(port, "127.0.0.1");
