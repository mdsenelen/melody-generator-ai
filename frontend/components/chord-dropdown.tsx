"use client";

import { useEffect, useState } from "react";

import { requestJson } from "../app/lib/request";

const FALLBACK_CHORDS = [
  "C",
  "Cm",
  "D",
  "Dm",
  "E",
  "Em",
  "F",
  "G",
  "Am",
  "Bm",
  "G7",
  "Cmaj7",
  "Am7",
  "Dm7",
];

type ChordDropdownProps = {
  value: string;
  onChange: (value: string) => void;
  label?: string;
  className?: string;
};

function ChordDropdown({
  value,
  onChange,
  label,
  className = "",
}: ChordDropdownProps) {
  const [chords, setChords] = useState<string[]>(FALLBACK_CHORDS);

  useEffect(() => {
    let cancelled = false;

    async function loadChords() {
      try {
        const data = await requestJson<{ chords?: string[] }>("/api/chords", {
          expectedContentType: "application/json",
        });
        if (!cancelled && Array.isArray(data.chords) && data.chords.length > 0) {
          setChords(data.chords);
        }
      } catch {
        if (!cancelled) {
          setChords(FALLBACK_CHORDS);
        }
      }
    }

    loadChords();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <label className={`flex flex-col gap-2 text-sm text-gray-300 ${className}`}>
      {label ? <span className="font-medium text-gray-200">{label}</span> : null}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="rounded-2xl border border-white/10 bg-gray-900 px-3 py-2 text-white outline-none transition focus:border-purple-400"
      >
        {chords.map((chord) => (
          <option key={chord} value={chord}>
            {chord}
          </option>
        ))}
      </select>
    </label>
  );
}

export default ChordDropdown;
