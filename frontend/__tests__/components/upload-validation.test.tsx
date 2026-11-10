import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";

import { UploadButton } from "../../components/upload-button";
import { MAX_UPLOAD_BYTES } from "../../src/domain/audioFile";
import { audioFile, emptyFile } from "../../src/test/fakes/files";
import { BACKEND_API, server } from "../../src/test/msw/server";

// A dropped file skips the input's `accept` filter, so these drive the
// input the way a drop does (applyAccept: false), not the way the picker does.
function setup() {
  const requests: string[] = [];
  server.use(
    http.all("*", ({ request }) => {
      requests.push(`${request.method} ${request.url}`);
      return HttpResponse.json({ id: "u1", filename: "upload_u1.wav" });
    }),
  );
  const onUploadSuccess = jest.fn();
  const onUploadError = jest.fn();
  render(<UploadButton onUploadSuccess={onUploadSuccess} onUploadError={onUploadError} />);
  const user = userEvent.setup({ applyAccept: false });
  const input = screen.getByLabelText(/upload audio file/i);
  return { user, input, requests, onUploadSuccess, onUploadError };
}

describe("UploadButton file checks", () => {
  it.each([
    [
      "wrong MIME type",
      audioFile({ name: "photo.wav", type: "image/png" }),
      /isn't an audio file/i,
    ],
    [
      "wrong extension",
      audioFile({ name: "notes.txt", type: "text/plain" }),
      /isn't a supported audio format/i,
    ],
    [
      "oversized file",
      audioFile({ name: "long-take.wav", size: MAX_UPLOAD_BYTES + 1 }),
      /too large.*50 MB/i,
    ],
    ["zero-byte file", emptyFile("silence.wav"), /is empty/i],
  ])("rejects a %s without sending anything", async (_case, file, message) => {
    const { user, input, requests, onUploadSuccess, onUploadError } = setup();

    await user.upload(input, file);

    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(onUploadError).toHaveBeenCalledWith(expect.stringMatching(message));
    expect(onUploadSuccess).not.toHaveBeenCalled();
    expect(requests).toEqual([]);
  });

  it("still uploads a valid file", async () => {
    const { user, input, requests, onUploadSuccess } = setup();

    await user.upload(input, audioFile({ name: "riff.wav" }));

    expect(await screen.findByText(/upload complete/i)).toBeInTheDocument();
    expect(onUploadSuccess).toHaveBeenCalledWith(
      expect.objectContaining({ id: "u1", filename: "upload_u1.wav" }),
    );
    expect(requests).toContain(`POST ${BACKEND_API}/upload`);
  });
});
