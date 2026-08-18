"use client";

import { useEffect, useState } from "react";

import { ChordGraph } from "../../components/chord-graph";
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
  Rock: "border-red-400/40    bg-red-500/15    text-red-100",
  General: "border-gray-400/40   bg-gray-500/15   text-gray-100",
};

function GenreBadge({ genre }: { genre: string }) {
  const cls = GENRE_BADGE[genre] ?? GENRE_BADGE.General;
  return (
    <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${cls}`}>{genre}</span>
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
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    requestJson<{ progressions: Progression[] }>("/api/progressions", {
      expectedContentType: "application/json",
    })
      .then((data) => setProgressions(data.progressions ?? []))
      .catch(() => setProgressions([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {loading ? (
        <div className="flex justify-center py-20">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-purple-400 border-t-transparent" />
        </div>
      ) : (
        <div className="grid gap-6">
          {progressions.map((prog) => (
            <ProgressionCard key={prog.id} prog={prog} />
          ))}
        </div>
      )}
    </div>
  );
}
