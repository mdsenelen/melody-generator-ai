"use client";

import { useState } from "react";

import { ChordGraph } from "../../components/chord-graph";
import { Spinner } from "../../components/spinner";
import { GenreBadge } from "../../components/ui/badge";
import { Card } from "../../components/ui/card";
import { Pills } from "../../components/ui/pills";
import { Label } from "../../components/ui/text";
import { useProgressions } from "../../hooks/use-progressions";
import type { Progression } from "../../src/domain/contracts/progressions";
import { WAKING_STATUS_MESSAGE } from "../../src/domain/statusCopy";

const NO_PROGRESSIONS: Progression[] = [];

function ProgressionCard({ prog }: { prog: Progression }) {
  const subtitle =
    prog.song_title && prog.artist
      ? `${prog.song_title} · ${prog.artist}${prog.year ? ` · ${prog.year}` : ""}`
      : undefined;

  return (
    <ChordGraph
      title={prog.name}
      description={subtitle}
      progression={prog.chords}
      genreBadge={<GenreBadge genre={prog.genre} />}
    />
  );
}

export default function ListenProgressionsPage() {
  const library = useProgressions();
  const [activeGenre, setActiveGenre] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");
  const progressions = library.status === "ready" ? library.progressions : NO_PROGRESSIONS;

  const genres = ["All", "Pop", "Jazz", "Blues", "Classical", "Flamenco", "J-Pop"];
  const normalizedSearch = searchQuery.trim().toLowerCase();
  const visibleProgressions = progressions.filter((progression) => {
    const matchesGenre = activeGenre === "All" || progression.genre === activeGenre;
    const searchableText = [
      progression.name,
      progression.genre,
      progression.source,
      progression.song_title,
      progression.artist,
      ...progression.chords,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return matchesGenre && (!normalizedSearch || searchableText.includes(normalizedSearch));
  });

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <Label>Curated library</Label>
          <h1 className="font-display text-foreground mt-2 text-2xl font-light">Progressions</h1>
          <p className="text-muted-foreground mt-2 max-w-xl text-sm">
            Explore chord movements, hear them in context, and download the progression as MIDI.
          </p>
        </div>

        <label className="relative block w-full lg:max-w-[320px]">
          <span className="sr-only">Search progressions</span>
          <span className="text-muted-foreground pointer-events-none absolute inset-y-0 left-4 flex items-center">
            ⌕
          </span>
          <input
            type="search"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search by name, chord, artist..."
            className="border-border bg-card text-foreground placeholder:text-muted-foreground focus:border-primary/60 h-11 w-full rounded-[var(--radius)] border pr-4 pl-11 text-sm transition-colors outline-none"
          />
        </label>
      </div>

      <Pills
        options={genres}
        value={activeGenre}
        onChange={setActiveGenre}
        name="Filter by genre"
      />

      {library.status === "loading" || library.status === "waking" ? (
        <div className="flex justify-center py-20">
          <Spinner
            size="lg"
            label={
              library.status === "waking" ? WAKING_STATUS_MESSAGE : "Loading chord progressions"
            }
          />
        </div>
      ) : library.status === "failed" ? (
        <div className="rounded-[var(--radius)] border border-red-500/40 bg-[#120808] p-4 text-sm text-red-100">
          Couldn&apos;t load chord progressions: {library.error}
        </div>
      ) : progressions.length === 0 ? (
        <Card
          variant="muted"
          className="text-muted-foreground flex min-h-[200px] items-center justify-center border-dashed text-center"
        >
          No chord progressions available yet.
        </Card>
      ) : visibleProgressions.length === 0 ? (
        <Card
          variant="muted"
          className="text-muted-foreground flex min-h-[200px] items-center justify-center border-dashed text-center"
        >
          No progressions match the current filters.
        </Card>
      ) : (
        <div className="grid gap-5 xl:grid-cols-2">
          {visibleProgressions.map((prog) => (
            <ProgressionCard key={prog.id} prog={prog} />
          ))}
        </div>
      )}
    </div>
  );
}
