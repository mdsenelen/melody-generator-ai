import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";

import GenerateVariantsPage from "../../app/generate-variants/page";
import { useSessionStore } from "../../app/lib/session-store";
import { variantsResponse } from "../../src/test/fixtures/variants";
import { server } from "../../src/test/msw/server";

// A clip already transcribed on /analyse: generating from it sends one
// request, POST /api/generate-variants, and nothing else.
function rememberAnalysedClip() {
  useSessionStore.getState().setLastUpload({
    uploadId: "abc123",
    filename: "upload_abc123.wav",
    sourceName: "my-riff.wav",
    jobId: "job-1",
    transcription: { chords: [], key: "C major", moodLabel: "happy", pitchHistogram: [] },
  });
}

async function chooseClipAndGenerate(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: /use my last upload/i }));
  await user.click(screen.getByRole("button", { name: /generate variants/i }));
}

beforeEach(() => {
  useSessionStore.setState({ lastUpload: null });
  rememberAnalysedClip();
});

describe("Generate Variants: retry after failure", () => {
  it("clears the failure and shows the variants when the user generates again", async () => {
    let attempts = 0;
    server.use(
      http.post("/api/generate-variants", () => {
        attempts += 1;
        return attempts === 1
          ? HttpResponse.json({ detail: "The model crashed." }, { status: 500 })
          : HttpResponse.json(variantsResponse());
      }),
    );
    const user = userEvent.setup();
    render(<GenerateVariantsPage />);

    await chooseClipAndGenerate(user);
    expect(await screen.findByText("The model crashed.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /generate variants/i }));

    expect(await screen.findByRole("heading", { name: /rendered variants/i })).toBeInTheDocument();
    expect(screen.queryByText("The model crashed.")).not.toBeInTheDocument();
    expect(attempts).toBe(2);
  });
});

describe("Generate Variants: abort mid-generation", () => {
  it("aborts the in-flight request when the user leaves the page", async () => {
    const seen: { signal?: AbortSignal } = {};
    server.use(
      http.post("/api/generate-variants", ({ request }) => {
        seen.signal = request.signal;
        // Generation never finishes on its own: only an abort ends it.
        return new Promise<Response>(() => {});
      }),
    );
    const consoleError = jest.spyOn(console, "error");
    const user = userEvent.setup();
    const { unmount } = render(<GenerateVariantsPage />);

    await chooseClipAndGenerate(user);
    await waitFor(() => expect(seen.signal).toBeDefined());
    expect(screen.getByRole("button", { name: /generate variants/i })).toBeDisabled();
    expect(seen.signal?.aborted).toBe(false);

    unmount();

    await waitFor(() => expect(seen.signal?.aborted).toBe(true));
    // An abort is the user's choice, not a failure: nothing gets logged as one.
    expect(consoleError).not.toHaveBeenCalled();
    consoleError.mockRestore();
  });
});
