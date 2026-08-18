// components/training-progress.tsx
"use client";
import { useEffect, useState } from "react";

import { requestJson } from "../app/lib/request";

export function TrainingProgress() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const fetchProgress = async () => {
      try {
        const data = await requestJson<{ progress?: number }>("/api/training-progress", {
          expectedContentType: "application/json",
        });
        if (typeof data.progress === "number") {
          setProgress(data.progress);
        }
      } catch {
        return;
      }
    };

    const interval = setInterval(fetchProgress, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="mt-4">
      <div className="h-2 overflow-hidden rounded-full bg-gray-200">
        <div
          className="h-full bg-indigo-600 transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
      <p className="mt-1 text-xs text-gray-500">Training progress: {Math.round(progress)}%</p>
    </div>
  );
}
