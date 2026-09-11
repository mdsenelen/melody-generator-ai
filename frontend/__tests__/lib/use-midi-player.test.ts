import { act, renderHook, waitFor } from "@testing-library/react";
import { Midi } from "@tonejs/midi";

import { useMidiPlayer } from "../../hooks/use-midi-player";

// ---- Tone.js mock -----------------------------------------------------------
// The hook loads Tone via dynamic import(); jsdom has no AudioContext, so the
// whole module is faked. We assert on lifecycle calls, not on real audio.

const disposed = { part: 0, sampler: 0 };
let lastOnload: (() => void) | undefined;
const transport = {
  seconds: 0,
  start: jest.fn(),
  pause: jest.fn(),
  stop: jest.fn(),
  cancel: jest.fn(),
  position: 0,
};

jest.mock("tone", () => ({
  start: jest.fn().mockResolvedValue(undefined),
  getTransport: () => transport,
  Sampler: jest.fn().mockImplementation((opts: { onload?: () => void }) => {
    lastOnload = opts.onload;
    return {
      toDestination() {
        return this;
      },
      dispose() {
        disposed.sampler += 1;
      },
    };
  }),
  Part: jest.fn().mockImplementation(() => ({
    start() {
      return this;
    },
    dispose() {
      disposed.part += 1;
    },
  })),
}));

function midiBase64(): string {
  const midi = new Midi();
  const track = midi.addTrack();
  track.addNote({ midi: 60, time: 0, duration: 0.5, velocity: 0.8 });
  track.addNote({ midi: 64, time: 0.5, duration: 0.5, velocity: 0.8 });
  const bytes = midi.toArray();
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

beforeEach(() => {
  disposed.part = 0;
  disposed.sampler = 0;
  transport.seconds = 0;
  jest.clearAllMocks();
});

it("reports 'empty' when there is no MIDI", () => {
  const { result } = renderHook(() => useMidiPlayer(null));
  expect(result.current.state.status).toBe("empty");
});

it("goes idle -> loading -> playing on the first toggle, then pauses", async () => {
  const { result } = renderHook(() => useMidiPlayer(midiBase64()));
  expect(result.current.state.status).toBe("idle");

  act(() => {
    result.current.toggle();
  });
  expect(result.current.state.status).toBe("loading");

  // let the dynamic import + Tone.start() + Sampler construction settle
  await act(async () => {});
  await waitFor(() => expect(lastOnload).toBeDefined());
  // sampler finishes loading
  await act(async () => {
    lastOnload?.();
  });

  await waitFor(() => expect(result.current.state.status).toBe("playing"));
  expect(transport.start).toHaveBeenCalled();

  act(() => {
    result.current.toggle();
  });
  expect(result.current.state.status).toBe("paused");
  expect(transport.pause).toHaveBeenCalled();
});

it("disposes the Tone graph on unmount", async () => {
  const { result, unmount } = renderHook(() => useMidiPlayer(midiBase64()));

  act(() => {
    result.current.toggle();
  });
  await act(async () => {});
  await waitFor(() => expect(lastOnload).toBeDefined());
  await act(async () => {
    lastOnload?.();
  });
  await waitFor(() => expect(result.current.state.status).toBe("playing"));

  unmount();

  expect(disposed.part).toBeGreaterThan(0);
  expect(disposed.sampler).toBeGreaterThan(0);
  expect(transport.stop).toHaveBeenCalled();
});

it("surfaces a parse failure as an error state, without throwing", () => {
  const { result } = renderHook(() => useMidiPlayer(btoa("not a midi file at all")));
  expect(result.current.state.status).toBe("error");
});
