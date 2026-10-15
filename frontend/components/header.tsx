import Link from "next/link";

export default function Header() {
  return (
    <Link
      href="/"
      className="focus-visible:ring-ring flex items-center gap-2.5 rounded-full transition-opacity hover:opacity-80 focus-visible:ring-2 focus-visible:outline-none"
    >
      {/* Decorative: the wordmark beside it names the link. */}
      <img src="/logo.svg" alt="" width={28} height={28} className="h-7 w-7" />
      <span className="font-title text-foreground text-2xl leading-none">Melodia</span>
    </Link>
  );
}
