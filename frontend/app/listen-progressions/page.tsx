"use client";

import { useEffect, useState } from "react";

import { ChordGraph } from "../../components/chord-graph";
import { Spinner } from "../../components/spinner";
import { GenreBadge } from "../../components/ui/badge";
import { Card } from "../../components/ui/card";
import { Pills } from "../../components/ui/pills";
import { PageHeader } from "../../components/ui/text";
import { requestJson } from "../lib/request";

type Progression = {
  id: string;
  name: string;
  genre: string;
  source: string;
  chords: string[];
  song_title: string | null;
  artist: string | null;
  year: number | null;
};

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
  const [progressions, setProgressions] = useState<Progression[]>([]);
  const [activeGenre, setActiveGenre] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadProgressions() {
      try {
        const data = await requestJson<{ progressions: Progression[] }>("/api/progressions", {
          expectedContentType: "application/json",
        });
        if (!cancelled) {
          setProgressions(data.progressions ?? []);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(
            fetchError instanceof Error ? fetchError.message : "Failed to load progressions",
          );
          setProgressions([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadProgressions();
    return () => {
      cancelled = true;
    };
  }, []);

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
    <div className="space-y-10">
      <PageHeader
        eyebrow="Curated library"
        title="Progressions"
        lead="Explore chord movements, hear them in context, and download the progression as MIDI."
      >
        <label className="relative block w-full md:w-[320px]">
          <span className="sr-only">Search progressions</span>
          <span
            aria-hidden="true"
            className="text-muted-foreground pointer-events-none absolute inset-y-0 left-4 flex items-center"
          >
            ⌕
          </span>
          <input
            type="search"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search by name, chord, artist..."
            className="border-border-strong bg-card text-foreground placeholder:text-muted-foreground focus-visible:ring-ring h-11 w-full rounded-full border pr-4 pl-11 text-sm transition-colors outline-none focus-visible:ring-2"
          />
        </label>
      </PageHeader>

      <Pills
        options={genres}
        value={activeGenre}
        onChange={setActiveGenre}
        name="Filter by genre"
      />

      {loading ? (
        <div className="flex justify-center py-20">
          <Spinner size="lg" label="Loading chord progressions" />
        </div>
      ) : error ? (
        <div
          role="alert"
          className="border-danger/40 bg-danger/8 text-foreground rounded-[var(--radius)] border p-4 text-sm"
        >
          Couldn&apos;t load chord progressions: {error}
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
