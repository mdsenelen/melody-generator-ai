import Link from "next/link";

export default function Header() {
  return (
    <Link href="/" className="flex items-center gap-2.5 transition-opacity hover:opacity-75">
      <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
        <circle cx="11" cy="11" r="9.5" stroke="#AC20E8" strokeWidth="1.2" />
        <path d="M7 11c0-2.2 1.8-4 4-4" stroke="#AC20E8" strokeWidth="1.2" strokeLinecap="round" />
        <path d="M15 11c0 2.2-1.8 4-4 4" stroke="#AC20E8" strokeWidth="1.2" strokeLinecap="round" />
        <circle cx="11" cy="11" r="1.5" fill="#AC20E8" />
      </svg>
      <span className="font-display text-foreground text-base font-normal tracking-tight">
        Melodia
      </span>
    </Link>
  );
}
