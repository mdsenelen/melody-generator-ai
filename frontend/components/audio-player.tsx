// components/audio-player.tsx
"use client";
import { useState, useRef } from "react";

export function AudioPlayer({ audioUrl }: { audioUrl: string }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  const togglePlay = () => {
    if (isPlaying) {
      audioRef.current?.pause();
    } else {
      audioRef.current?.play();
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <div className="flex items-center gap-2">
      <button onClick={togglePlay} className="rounded-full bg-indigo-100 p-2 text-indigo-600">
        <span aria-hidden="true">{isPlaying ? "Pause" : "Play"}</span>
      </button>
      <audio ref={audioRef} src={audioUrl} />
    </div>
  );
}
