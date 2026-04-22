import Link from "next/link";

const links = [
  { href: "/", label: "Analyse" },
  { href: "/listen-progressions", label: "Progressions" },
  { href: "/choose-progression", label: "Compose" },
  { href: "/generate-variants", label: "Variants" },
];

export default function Nav() {
  return (
    <nav className="flex flex-wrap items-center gap-1.5">
      {links.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          className="rounded-full border border-white/10 bg-white/5 px-3.5 py-1.5 text-[13px] font-medium text-white/70 backdrop-blur-sm transition hover:border-white/20 hover:bg-white/10 hover:text-white"
        >
          {link.label}
        </Link>
      ))}
    </nav>
  );
}
