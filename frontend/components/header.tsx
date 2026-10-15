import Link from "next/link";

export default function Header() {
  return (
    <Link
      href="/"
      className="focus-visible:ring-ring flex items-center gap-2.5 rounded-full transition-opacity hover:opacity-80 focus-visible:ring-2 focus-visible:outline-none"
    >
      <svg
        width="24"
        height="24"
        viewBox="0 0 22 22"
        fill="none"
        aria-hidden="true"
        className="text-primary"
      >
        <circle cx="11" cy="11" r="9.5" stroke="currentColor" strokeWidth="1.2" />
        <path
          d="M7 11c0-2.2 1.8-4 4-4"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeLinecap="round"
        />
        <path
          d="M15 11c0 2.2-1.8 4-4 4"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeLinecap="round"
        />
        <circle cx="11" cy="11" r="1.5" fill="currentColor" />
      </svg>
      <span className="font-title text-foreground text-lg font-medium tracking-tight">Melodia</span>
    </Link>
  );
}
