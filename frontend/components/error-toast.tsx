// components/error-toast.tsx
"use client";
import { useEffect } from "react";

export function ErrorToast({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  useEffect(() => {
    // Long enough to actually read an error (vs. a success toast, which
    // can disappear quickly) — dismissing is still available via the ×.
    const timer = setTimeout(onDismiss, 20000);
    return () => clearTimeout(timer);
  }, [onDismiss]);

  return (
    <div className="fixed right-4 bottom-4 z-50 flex max-w-xs items-start gap-2 rounded-[var(--radius)] border border-red-500/50 bg-[#120808] p-4 text-red-100 shadow-[0_12px_32px_rgba(0,0,0,0.5)]">
      <span className="mt-0.5 flex-shrink-0 text-base font-bold">!</span>
      <div className="flex-1">
        <p className="text-sm">{message}</p>
      </div>
      <button onClick={onDismiss} className="text-red-300 hover:text-white">
        <span aria-hidden="true">×</span>
      </button>
    </div>
  );
}
