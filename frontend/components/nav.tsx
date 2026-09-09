import Link from "next/link";

const links = [
  { href: "/analyse", label: "Analyse" },
  { href: "/listen-progressions", label: "Progressions" },
  { href: "/choose-progression", label: "Compose" },
  { href: "/generate-variants", label: "Variants" },
];

export default function Nav() {
  return (
    <nav className="flex flex-wrap items-center gap-2 text-[11px] font-medium tracking-[0.18em] text-white/55 uppercase">
      {links.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          className="rounded-full border border-transparent px-2.5 py-1.5 transition hover:text-[#d7b9ff]"
        >
          {link.label}
        </Link>
      ))}
    </nav>
  );
}
