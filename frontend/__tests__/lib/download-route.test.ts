/**
 * @jest-environment node
 */
// The /api/download/[filename] route is server code (NextRequest needs the
// node environment's web globals). FastAPI behind it is served by MSW.
import { http, HttpResponse } from "msw";
import { NextRequest } from "next/server";

import { GET } from "../../app/api/download/[filename]/route";
import { BACKEND_API, server } from "../../src/test/msw/server";

const MIDI_BYTES = new Uint8Array([0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6]); // "MThd" header
const WAV_BYTES = new Uint8Array([0x52, 0x49, 0x46, 0x46, 0x24, 0, 0, 0]); // "RIFF"

function backendServes(filename: string, bytes: Uint8Array, type: string) {
  server.use(
    http.get(`${BACKEND_API}/download/:filename`, ({ params }) =>
      params.filename === filename
        ? new HttpResponse(bytes, {
            headers: {
              "content-type": type,
              "content-disposition": `attachment; filename="${filename}"`,
            },
          })
        : HttpResponse.json({ detail: "File not found" }, { status: 404 }),
    ),
  );
}

async function download(filename: string) {
  const request = new NextRequest(`http://localhost/api/download/${encodeURIComponent(filename)}`);
  return GET(request, { params: Promise.resolve({ filename }) });
}

describe("GET /api/download/[filename]", () => {
  it("returns a MIDI file as an audio/midi blob named after the file", async () => {
    backendServes("variant_1.mid", MIDI_BYTES, "audio/midi");

    const response = await download("variant_1.mid");

    expect(response.status).toBe(200);
    expect(response.headers.get("content-disposition")).toBe(
      'attachment; filename="variant_1.mid"',
    );
    const blob = await response.blob();
    expect(blob.type).toBe("audio/midi");
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(MIDI_BYTES);
  });

  it("returns a WAV file as an audio/wav blob named after the file", async () => {
    backendServes("progression.wav", WAV_BYTES, "audio/wav");

    const response = await download("progression.wav");

    expect(response.headers.get("content-disposition")).toBe(
      'attachment; filename="progression.wav"',
    );
    const blob = await response.blob();
    expect(blob.type).toBe("audio/wav");
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(WAV_BYTES);
  });

  it("asks the backend for the exact file, even with # or spaces in its name", async () => {
    backendServes("take #2.mid", MIDI_BYTES, "audio/midi");

    const response = await download("take #2.mid");

    expect(response.status).toBe(200);
  });

  it("passes a missing file through as a 404, not an empty download", async () => {
    backendServes("variant_1.mid", MIDI_BYTES, "audio/midi");

    const response = await download("nope.mid");

    expect(response.status).toBe(404);
    expect(response.headers.get("content-disposition")).toBeNull();
  });
});
