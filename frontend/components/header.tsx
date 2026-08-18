import Link from "next/link";

export default function Header() {
  return (
    <Link href="/" className="flex items-center gap-2.5 transition-opacity hover:opacity-75">
      <img src="/logo.svg" alt="MelodyAI" className="h-7 w-auto" />
      <span className="text-[15px] font-semibold tracking-tight text-white">MelodyAI</span>
    </Link>
  );
}
