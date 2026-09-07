"use client";

import { useEffect, useState } from "react";

import { ChordGraph } from "../../components/chord-graph";
import { Spinner } from "../../components/spinner";
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

const GENRE_BADGE: Record<string, string> = {
  Jazz: "border-amber-400/40  bg-amber-500/15  text-amber-100",
  Blues: "border-blue-400/40   bg-blue-500/15   text-blue-100",
  Pop: "border-purple-400/40 bg-purple-500/15 text-purple-100",
  Classical: "border-emerald-400/40 bg-emerald-500/15 text-emerald-100",
  Flamenco: "border-orange-400/40 bg-orange-500/15 text-orange-100",
  "J-Pop": "border-fuchsia-400/40 bg-fuchsia-500/15 text-fuchsia-100",
  Rock: "border-red-400/40    bg-red-500/15    text-red-100",
  General: "border-gray-400/40   bg-gray-500/15   text-gray-100",
};

function GenreBadge({ genre }: { genre: string }) {
  const cls = GENRE_BADGE[genre] ?? GENRE_BADGE.General;
  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold tracking-[0.16em] uppercase ${cls}`}
    >
      {genre}
    </span>
  );
}

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
    <div className="space-y-8 pb-8">
      <div className="flex flex-col gap-5 border-b border-white/10 pb-6 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-[11px] font-medium tracking-[0.24em] text-[#bf9bff] uppercase">
            Curated library
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-4xl">
            Progressions
          </h1>
          <p className="mt-2 max-w-xl text-sm text-white/55">
            Explore chord movements, hear them in context, and download the progression as MIDI.
          </p>
        </div>

        <label className="relative block w-full lg:max-w-[320px]">
          <span className="sr-only">Search progressions</span>
          <span className="pointer-events-none absolute inset-y-0 left-4 flex items-center text-white/40">
            ⌕
          </span>
          <input
            type="search"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search by name, chord, artist..."
            className="h-11 w-full rounded-xl border border-white/12 bg-[rgba(12,16,25,0.72)] pr-4 pl-11 text-sm text-white transition outline-none placeholder:text-white/35 focus:border-[#9b6cff]/70 focus:ring-2 focus:ring-[#8b5cf6]/15"
          />
        </label>
      </div>

      <div className="flex flex-wrap gap-2" aria-label="Filter by genre">
        {genres.map((genre) => {
          const isActive = activeGenre === genre;
          return (
            <button
              key={genre}
              type="button"
              onClick={() => setActiveGenre(genre)}
              aria-pressed={isActive}
              className={`rounded-lg border px-4 py-2 text-xs font-medium transition ${
                isActive
                  ? "border-[#a879ff]/70 bg-[#8b5cf6]/25 text-white shadow-[0_0_18px_rgba(139,92,246,0.14)]"
                  : "border-white/10 bg-white/[0.035] text-white/55 hover:border-white/25 hover:text-white"
              }`}
            >
              {genre}
            </button>
          );
        })}
      </div>

      {loading ? (
        <div className="flex justify-center py-20">
          <Spinner size="lg" label="Loading chord progressions" />
        </div>
      ) : error ? (
        <div className="rounded-2xl border border-red-500/40 bg-red-950/40 p-4 text-sm text-red-100">
          Couldn&apos;t load chord progressions: {error}
        </div>
      ) : progressions.length === 0 ? (
        <div className="flex min-h-[200px] items-center justify-center rounded-3xl border border-dashed border-white/15 bg-white/5 p-8 text-center text-white/65">
          No chord progressions available yet.
        </div>
      ) : visibleProgressions.length === 0 ? (
        <div className="flex min-h-[200px] items-center justify-center rounded-3xl border border-dashed border-white/15 bg-white/5 p-8 text-center text-white/65">
          No progressions match the current filters.
        </div>
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
