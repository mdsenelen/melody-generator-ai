import Link from "next/link";

const links = [
  { href: "/", label: "Analyse Audio" },
  { href: "/listen-progressions", label: "Listen Progressions" },
  { href: "/choose-progression", label: "Choose Progression" },
  { href: "/generate-variants", label: "Variants" },
];

const Nav = () => (
  <nav className="flex flex-wrap items-center gap-2 text-sm font-medium text-gray-200">
    {links.map((link) => (
      <Link
        key={link.href}
        href={link.href}
        className="rounded-full border border-white/15 bg-white/5 px-4 py-2 backdrop-blur-sm transition hover:border-purple-400/60 hover:bg-purple-500/10 hover:text-white"
      >
        {link.label}
      </Link>
    ))}
  </nav>
);

export default Nav;
