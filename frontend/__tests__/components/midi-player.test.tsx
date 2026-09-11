import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { MidiPlayer } from "../../components/midi-player";
import type { MidiPlayer as MidiPlayerApi, MidiPlayerState } from "../../hooks/use-midi-player";

const toggle = jest.fn();
const stop = jest.fn();
let mockState: MidiPlayerState = { status: "idle", durationSec: 4 };

jest.mock("../../hooks/use-midi-player", () => ({
  useMidiPlayer: (): MidiPlayerApi => ({ state: mockState, toggle, stop }),
}));

function renderWith(state: MidiPlayerState) {
  mockState = state;
  return render(<MidiPlayer midiB64="AAA=" />);
}

beforeEach(() => {
  toggle.mockClear();
  stop.mockClear();
});

it("shows a Play button when idle and calls toggle on click", async () => {
  const user = userEvent.setup();
  renderWith({ status: "idle", durationSec: 4 });

  const button = screen.getByRole("button", { name: /play/i });
  expect(button).toBeEnabled();
  expect(button).toHaveAttribute("aria-pressed", "false");

  await user.click(button);
  expect(toggle).toHaveBeenCalledTimes(1);
});

it("shows a labelled loading state with the button disabled", () => {
  renderWith({ status: "loading", durationSec: 4 });

  expect(screen.getByRole("button", { name: /loading|play/i })).toBeDisabled();
  expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
});

it("becomes a Pause button while playing", () => {
  renderWith({ status: "playing", durationSec: 4, positionSec: 1 });

  const button = screen.getByRole("button", { name: /pause/i });
  expect(button).toHaveAttribute("aria-pressed", "true");
});

it("shows an inline message on error, without throwing", () => {
  renderWith({ status: "error", message: "This melody couldn't be read for playback." });

  expect(screen.getByText(/couldn't be read for playback/i)).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /play|pause/i })).not.toBeInTheDocument();
});

it("shows an empty state when there is nothing to play", () => {
  renderWith({ status: "empty" });

  expect(screen.getByText(/no audio to play/i)).toBeInTheDocument();
});

it("renders a progress bar that does not animate under reduced motion", () => {
  renderWith({ status: "playing", durationSec: 4, positionSec: 2 });

  const bar = screen.getByRole("progressbar");
  expect(bar).toHaveAttribute("aria-valuenow", "2");
  expect(bar).toHaveAttribute("aria-valuemax", "4");
  // the moving fill opts out of transitions when the user asked for less motion
  expect(bar.querySelector(".motion-reduce\\:transition-none")).not.toBeNull();
});
