import Link from "next/link";

export default function Header() {
  return (
    <Link href="/" className="flex items-center gap-2.5 transition-opacity hover:opacity-75">
      <img src="/melodyai-logo.png" alt="" width={22} height={22} />
      <span className="font-display text-foreground text-base font-normal tracking-tight">
        Melodia
      </span>
    </Link>
  );
}
