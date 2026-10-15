"use client";

import { useState } from "react";

import { ChordDiagram } from "../../components/chord-diagram";
import { ChordGraph } from "../../components/chord-graph";
import { ChordPicker } from "../../components/chord-picker";
import { KeyboardStrip } from "../../components/keyboard-strip";
import { PlannedBadge } from "../../components/ui/badge";
import { cn } from "../../components/ui/cn";
import { Label, PageHeader } from "../../components/ui/text";
import { chordPitchClasses } from "../../data/chords";
import { NOTE_NAMES } from "../../utils/pitch";

const PLANNED_FEATURES = [
  "Add / remove chord slots (up to 8)",
  "Instrument selector (Piano · Guitar · Strings · Bass)",
  "Voice leading optimiser",
  "Export as chord sheet PDF",
];

export default function ChooseProgressionPage() {
  const [progression, setProgression] = useState<(string | null)[]>([null, null, null, null]);
  // The slot the readout + keyboard follow: whichever chord was edited last.
  const [focusIndex, setFocusIndex] = useState<number | null>(null);

  const updateChord = (index: number, chord: string | null) => {
    setProgression((current) =>
      current.map((value, currentIndex) => (currentIndex === index ? chord : value)),
    );
    setFocusIndex(chord ? index : null);
  };

  const filled = progression.filter((chord): chord is string => Boolean(chord));
  const focusChord = focusIndex !== null ? progression[focusIndex] : null;
  const focusPitchClasses = focusChord ? chordPitchClasses(focusChord) : [];

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Compose"
        title="Build your progression"
        lead="Choose up to four chords, hear the movement, then use it as a scaffold for generation."
      />

      <section
        aria-label="Progression"
        className="border-border bg-card overflow-hidden rounded-[var(--radius-lg)] border shadow-[var(--shadow-panel)]"
      >
        <div className="border-border bg-raised/60 flex items-center justify-between gap-3 border-b px-6 py-3">
          <Label>Progression</Label>
          <span className="text-primary-bright font-mono text-[13px] font-medium">
            {filled.length}/4 selected
          </span>
        </div>

        <div className="grid gap-px bg-[var(--color-border)] sm:grid-cols-2 xl:grid-cols-4">
          {progression.map((chord, index) => (
            <div
              key={`slot-${index}`}
              className={cn(
                "bg-card space-y-4 p-5 transition-colors",
                focusIndex === index && chord && "bg-primary-deep/40",
              )}
            >
              <Label>Chord {index + 1}</Label>
              <ChordPicker
                value={chord}
                onChange={(value) => updateChord(index, value)}
                placeholder={`Chord ${index + 1}`}
              />
              {chord ? (
                <div className="flex justify-center py-1">
                  <ChordDiagram chord={chord} />
                </div>
              ) : (
                <div className="border-border flex h-24 items-center justify-center rounded-[var(--radius)] border border-dashed">
                  <span className="text-muted-foreground text-sm">Add a chord</span>
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="border-border grid gap-6 border-t p-6 md:grid-cols-[220px_1fr] md:items-center">
          <div aria-live="polite">
            <Label>Now showing</Label>
            <p
              data-testid="chord-readout"
              className={cn(
                "font-display mt-2 text-5xl leading-none font-light",
                focusChord ? "text-foreground" : "text-muted-foreground",
              )}
            >
              {focusChord ?? "—"}
            </p>
            <p className="text-secondary-foreground mt-3 font-mono text-[13px]">
              {focusPitchClasses.length > 0
                ? focusPitchClasses.map((pc) => NOTE_NAMES[pc]).join(" · ")
                : "Pick a chord to see its notes"}
            </p>
          </div>
          <KeyboardStrip pitchClasses={focusPitchClasses} />
        </div>
      </section>

      <ChordGraph
        title="Preview progression"
        description={
          filled.length < 2
            ? "Add at least 2 chords to preview"
            : "Adjust tempo and listen to the selected chord movement."
        }
        progression={filled}
        canPlay={filled.length >= 2}
      />

      <section aria-labelledby="planned-heading" className="border-border border-t pt-10">
        <div className="flex items-center justify-between gap-4">
          <div>
            <Label>Coming next</Label>
            <h2 id="planned-heading" className="font-title text-foreground mt-2 text-2xl">
              More composition controls
            </h2>
          </div>
        </div>
        <ul className="mt-6 grid gap-3 sm:grid-cols-2">
          {PLANNED_FEATURES.map((feature) => (
            <li
              key={feature}
              className="border-border text-secondary-foreground flex items-center justify-between gap-3 rounded-[var(--radius)] border px-4 py-3 text-sm"
            >
              <span>{feature}</span>
              <PlannedBadge />
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
