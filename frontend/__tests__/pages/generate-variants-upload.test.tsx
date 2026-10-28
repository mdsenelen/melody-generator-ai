import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import GenerateVariantsPage from "../../app/generate-variants/page";

// Stands in for the real UploadButton, which has already sent the file
// straight to the backend by the time it reports success.
const UPLOADED = new File([new Uint8Array(6 * 1024 * 1024)], "big-song.mp3", {
  type: "audio/mpeg",
});

jest.mock("../../components/upload-button", () => ({
  UploadButton: ({
    onUploadSuccess,
  }: {
    onUploadSuccess: (payload: {
      id: string;
      filename: string;
      file: File;
      pickedAt: number;
    }) => void;
  }) => (
    <button
      type="button"
      onClick={() =>
        onUploadSuccess({
          id: "0123456789abcdef0123456789abcdef",
          filename: "upload_0123456789abcdef0123456789abcdef.mp3",
          file: UPLOADED,
          pickedAt: 0,
        })
      }
    >
      Fake upload
    </button>
  ),
}));

jest.mock("../../hooks/use-midi-player", () => ({
  useMidiPlayer: () => ({
    state: { status: "idle", durationSec: 3 },
    toggle: jest.fn(),
    stop: jest.fn(),
  }),
}));

it("references an uploaded file instead of re-sending its bytes through the Vercel proxy", async () => {
  // /api/generate-variants goes through a Vercel function, which rejects
  // bodies over ~4.5 MB with a bare 413. The upload is already on the
  // backend, so only its reference needs to travel.
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    status: 200,
    headers: new Headers({ "content-type": "application/json" }),
    text: async () =>
      JSON.stringify({
        n_variants: 1,
        temperatures: [0.7],
        mood_idx: 0,
        mood_label: "happy",
        model_status: {
          cvae: { path: "cvae", exists: true, size_mb: 1, loaded: true },
          iddm_ppo: { path: "iddm", exists: true, size_mb: 1, loaded: true },
          device: "cpu",
          load_error: null,
          fluidsynth_available: true,
        },
        variants: [],
        job_id: "job-upload",
      }),
  }) as unknown as typeof fetch;

  const user = userEvent.setup();
  render(<GenerateVariantsPage />);
  await user.click(screen.getByRole("button", { name: /fake upload/i }));
  await user.click(screen.getByRole("button", { name: /generate variants/i }));

  const body = (global.fetch as jest.Mock).mock.calls[0][1].body as FormData;
  expect(body.get("file")).toBeNull();
  expect(body.get("upload_id")).toBe("0123456789abcdef0123456789abcdef");
  expect(body.get("filename")).toBe("upload_0123456789abcdef0123456789abcdef.mp3");
});
