import { render, screen } from "@testing-library/react";

import { NoteRoll } from "../../components/note-roll";

type Ctx = Record<string, jest.Mock | string | number>;

describe("NoteRoll", () => {
  let fillRect: jest.Mock;

  beforeEach(() => {
    fillRect = jest.fn();
    // jsdom has no layout; give the canvas a width so the waveform has columns.
    jest.spyOn(HTMLCanvasElement.prototype, "clientWidth", "get").mockReturnValue(400);
    const ctx: Ctx = {
      fillRect,
      clearRect: jest.fn(),
      fillText: jest.fn(),
      setTransform: jest.fn(),
    };
    jest
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
  });

  afterEach(() => jest.restoreAllMocks());

  it("describes itself to assistive tech with the note count", () => {
    render(
      <NoteRoll
        notes={[{ start: 0, end: 1, pitch: 60, velocity: 90 }]}
        durationSec={10}
        peaks={null}
      />,
    );
    expect(screen.getByRole("img", { name: /1 detected notes/i })).toBeInTheDocument();
  });

  it("paints without a waveform and without notes (empty states)", () => {
    render(<NoteRoll notes={[]} durationSec={10} peaks={null} />);
    expect(fillRect).toHaveBeenCalled();
  });

  it("paints more with a waveform than without", () => {
    render(<NoteRoll notes={[]} durationSec={10} peaks={null} />);
    const without = fillRect.mock.calls.length;
    fillRect.mockClear();
    render(<NoteRoll notes={[]} durationSec={10} peaks={Float32Array.from([0.5, 1, 0.2])} />);
    expect(fillRect.mock.calls.length).toBeGreaterThan(without);
  });
});
