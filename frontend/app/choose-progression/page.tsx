"use client";

import { useState } from "react";

import { ChordDiagram } from "../../components/chord-diagram";
import { ChordGraph } from "../../components/chord-graph";
import { ChordPicker } from "../../components/chord-picker";
import { Card } from "../../components/ui/card";
import { PlannedBadge } from "../../components/ui/badge";
import { Label } from "../../components/ui/text";

const PLANNED_FEATURES = [
  "Add / remove chord slots (up to 8)",
  "Instrument selector (Piano · Guitar · Strings · Bass)",
  "Voice leading optimiser",
  "Export as chord sheet PDF",
];

export default function ChooseProgressionPage() {
  const [progression, setProgression] = useState<(string | null)[]>([null, null, null, null]);

  const updateChord = (index: number, chord: string | null) => {
    setProgression((current) =>
      current.map((value, currentIndex) => (currentIndex === index ? chord : value)),
    );
  };

  const filled = progression.filter((chord): chord is string => Boolean(chord));

  return (
    <div className="space-y-8">
      <div>
        <Label>Compose</Label>
        <h1 className="font-display text-foreground mt-2 text-3xl font-light">
          Build your progression
        </h1>
        <p className="text-muted-foreground mt-2 max-w-xl text-sm">
          Choose up to four chords and preview the movement before generating your melody.
        </p>
      </div>

      <Card>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {progression.map((chord, index) => (
            <div key={`slot-${index}`} className="space-y-3">
              <Label>Chord {index + 1}</Label>
              <ChordPicker
                value={chord}
                onChange={(value) => updateChord(index, value)}
                placeholder={`Chord ${index + 1}`}
              />
              {chord ? (
                <div className="flex justify-center py-2">
                  <ChordDiagram chord={chord} />
                </div>
              ) : (
                <div className="flex h-20 items-center justify-center">
                  <span className="text-border-strong font-mono text-[10px] tracking-widest uppercase">
                    Empty
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
        <div className="border-border mt-6 flex flex-wrap items-center justify-between gap-3 border-t pt-5">
          <p className="text-muted-foreground text-sm">
            {filled.length < 2
              ? "Add at least 2 chords to preview"
              : "Your progression is ready to preview"}
          </p>
          <span className="text-muted-foreground font-mono text-[10px] tracking-widest uppercase">
            {filled.length}/4 selected
          </span>
        </div>
      </Card>

      <ChordGraph
        title="Preview progression"
        description="Adjust tempo and listen to the selected chord movement."
        progression={filled}
        canPlay={filled.length >= 2}
      />

      <Card variant="muted">
        <div className="flex items-center justify-between gap-4">
          <div>
            <Label>Additional features</Label>
            <p className="text-muted-foreground mt-2 text-sm">
              More composition controls are on the way.
            </p>
          </div>
          <PlannedBadge />
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          {PLANNED_FEATURES.map((feature) => (
            <div
              key={feature}
              className="border-border text-muted-foreground flex items-center justify-between gap-3 rounded-[var(--radius)] border px-4 py-3 text-sm"
            >
              <span>{feature}</span>
              <PlannedBadge />
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
