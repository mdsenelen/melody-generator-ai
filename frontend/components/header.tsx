import Link from "next/link";
import React from "react";

export default function Header() {
  return (
    <div className="flex items-center">
      <Link href="/" className="flex items-center gap-3 transition-opacity hover:opacity-90">
        <img src="/logo.svg" alt="Logo" className="h-8 w-auto" />
        <div>
          <p className="text-lg font-bold tracking-wide text-white">MelodyAI</p>
          <p className="text-xs uppercase tracking-[0.24em] text-white/60"></p>
        </div>
      </Link>
    </div>
  );
}
