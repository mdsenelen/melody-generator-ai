// components/training-progress.tsx
'use client'
import { useEffect, useState } from 'react'

export function TrainingProgress() {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const fetchProgress = async () => {
      const res = await fetch('/api/training-progress')
      const data = await res.json()
      setProgress(data.progress)
    }
    
    const interval = setInterval(fetchProgress, 3000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="mt-4">
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div 
          className="h-full bg-indigo-600 transition-all duration-300" 
          style={{ width: `${progress}%` }}
        />
      </div>
      <p className="text-xs text-gray-500 mt-1">
        Training progress: {Math.round(progress)}%
      </p>
    </div>
  )
}