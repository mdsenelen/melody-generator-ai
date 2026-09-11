"use client";

import { Spinner } from "./spinner";
import { useMidiPlayer } from "../hooks/use-midi-player";

function formatTime(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(whole / 60);
  const secs = whole % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

type MidiPlayerProps = {
  midiB64: string | null;
  className?: string;
};

/**
 * Plays a generated melody in the browser (Tone.js, piano samples). The server
 * never renders audio for variants -- this is the only playback path.
 */
export function MidiPlayer({ midiB64, className = "" }: MidiPlayerProps) {
  const { state, toggle } = useMidiPlayer(midiB64);

  if (state.status === "empty") {
    return (
      <p className={`text-muted-foreground text-sm ${className}`}>
        No audio to play for this melody.
      </p>
    );
  }

  if (state.status === "error") {
    return (
      <p className={`text-sm text-amber-400/80 ${className}`}>
        {state.message} The MIDI download still works.
      </p>
    );
  }

  const isPlaying = state.status === "playing";
  const isLoading = state.status === "loading";
  const positionSec =
    state.status === "playing" || state.status === "paused" ? state.positionSec : 0;
  const durationSec = state.durationSec;
  const pct = durationSec > 0 ? Math.min(100, (positionSec / durationSec) * 100) : 0;

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <button
        type="button"
        onClick={toggle}
        disabled={isLoading}
        aria-pressed={isPlaying}
        aria-label={isPlaying ? "Pause melody" : "Play melody"}
        className="border-primary/60 bg-primary/20 text-foreground hover:bg-primary/30 focus-visible:ring-primary/60 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-full border transition-colors focus-visible:ring-2 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-60"
      >
        {isLoading ? (
          <Spinner size="sm" label="Loading piano" className="[&>span:last-child]:sr-only" />
        ) : isPlaying ? (
          <PauseIcon />
        ) : (
          <PlayIcon />
        )}
      </button>

      <div className="min-w-0 flex-1">
        <div
          role="progressbar"
          aria-valuenow={Math.round(positionSec)}
          aria-valuemin={0}
          aria-valuemax={Math.round(durationSec)}
          aria-label="Playback position"
          className="bg-border h-px w-full overflow-hidden"
        >
          <div
            className="bg-primary h-full transition-[width] duration-200 ease-linear motion-reduce:transition-none"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="text-muted-foreground mt-1 font-mono text-[11px] tabular-nums">
          {formatTime(positionSec)} / {formatTime(durationSec)}
        </p>
      </div>
    </div>
  );
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="h-4 w-4 fill-current">
      <path d="M4 2.5v11l9-5.5-9-5.5Z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="h-4 w-4 fill-current">
      <path d="M4 2.5h3v11H4v-11Zm5 0h3v11H9v-11Z" />
    </svg>
  );
}
