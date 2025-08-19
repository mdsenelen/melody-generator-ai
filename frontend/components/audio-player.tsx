// components/audio-player.tsx
'use client'
import { useState, useRef } from 'react'
import { FiPlay, FiPause } from 'react-icons/fi'

export function AudioPlayer({ audioUrl }: { audioUrl: string }) {
  const [isPlaying, setIsPlaying] = useState(false)
  const audioRef = useRef<HTMLAudioElement>(null)

  const togglePlay = () => {
    if (isPlaying) {
      audioRef.current?.pause()
    } else {
      audioRef.current?.play()
    }
    setIsPlaying(!isPlaying)
  }

  return (
    <div className="flex items-center gap-2">
      <button 
        onClick={togglePlay}
        className="p-2 rounded-full bg-indigo-100 text-indigo-600"
      >
        {isPlaying ? <FiPause /> : <FiPlay />}
      </button>
      <audio ref={audioRef} src={audioUrl} />
    </div>
  )
}