"use client";

import { useState } from "react";

import ChordDropdown from "../../components/chord-dropdown";
import { ChordGraph } from "../../components/chord-graph";

export default function ChooseProgressionPage() {
  const [progression, setProgression] = useState(["", "", "", ""]);

  const updateChord = (index: number, chord: string) => {
    setProgression((current) =>
      current.map((value, currentIndex) => (currentIndex === index ? chord : value)),
    );
  };

  return (
    <div className="space-y-8 pb-8">
      <div>
        <p className="text-[11px] font-medium tracking-[0.24em] text-[#bf9bff] uppercase">
          Compose
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-4xl">
          Build your progression
        </h1>
        <p className="mt-2 max-w-xl text-sm text-white/55">
          Choose up to four chords and preview the movement before generating your melody.
        </p>
      </div>

      <section className="rounded-[1.5rem] border border-white/10 bg-[rgba(17,22,32,0.74)] p-5 shadow-[0_12px_28px_rgba(2,6,23,0.2)] backdrop-blur-md sm:p-6">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {progression.map((chord, index) => (
            <ChordDropdown
              key={`slot-${index}`}
              label={`Chord ${index + 1}`}
              value={chord}
              onChange={(value) => updateChord(index, value)}
            />
          ))}
        </div>
        <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-white/10 pt-5">
          <p className="text-sm text-white/45">
            {progression.filter(Boolean).length < 2
              ? "Add at least 2 chords to preview"
              : "Your progression is ready to preview"}
          </p>
          <span className="text-[10px] font-semibold tracking-[0.2em] text-white/35 uppercase">
            {progression.filter(Boolean).length}/4 selected
          </span>
        </div>
      </section>

      <ChordGraph
        title="Preview progression"
        description="Adjust tempo and listen to the selected chord movement."
        progression={progression.filter(Boolean)}
        canPlay={progression.filter(Boolean).length >= 2}
      />

      <section className="rounded-[1.5rem] border border-white/10 bg-[rgba(17,22,32,0.58)] p-5 backdrop-blur-md sm:p-6">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-[10px] font-semibold tracking-[0.22em] text-white/40 uppercase">
              Additional features
            </p>
            <p className="mt-2 text-sm text-white/50">More composition controls are on the way.</p>
          </div>
          <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[10px] font-semibold tracking-[0.16em] text-white/40 uppercase">
            Planned
          </span>
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          {[
            "Add / remove chord slots (up to 8)",
            "Instrument selector (Piano · Guitar · Strings · Bass)",
            "Voice leading optimiser",
            "Export as chord sheet PDF",
          ].map((feature) => (
            <div
              key={feature}
              className="flex items-center justify-between gap-3 rounded-xl border border-white/8 bg-white/[0.03] px-4 py-3 text-sm text-white/55"
            >
              <span>{feature}</span>
              <span className="shrink-0 text-[10px] font-semibold tracking-[0.15em] text-[#bf9bff]/70 uppercase">
                Planned
              </span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
