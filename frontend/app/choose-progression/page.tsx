"use client";

import { useState } from "react";

import ChordDropdown from "../../components/chord-dropdown";
import { ChordGraph } from "../../components/chord-graph";

export default function ChooseProgressionPage() {
  const [progression, setProgression] = useState(["C", "G", "Am", "F"]);

  const updateChord = (index: number, chord: string) => {
    setProgression((current) =>
      current.map((value, currentIndex) => (currentIndex === index ? chord : value)),
    );
  };

  return (
    <div className="space-y-6">
      <section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
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
      </section>

      <ChordGraph
        title="Custom progression"
        description="Render the progression below with the selected instrument and BPM."
        progression={progression}
      />
    </div>
  );
}
