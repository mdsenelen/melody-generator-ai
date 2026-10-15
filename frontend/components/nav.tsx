"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/guide", label: "User's guide" },
  { href: "/analyse", label: "Analyse" },
  { href: "/listen-progressions", label: "Progressions" },
  { href: "/choose-progression", label: "Compose" },
  { href: "/generate-variants", label: "Variants" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Main"
      className="border-border bg-card/80 flex max-w-full items-center gap-1 overflow-x-auto rounded-full border p-1 shadow-[var(--shadow-panel)]"
    >
      {links.map((link) => {
        const active = pathname === link.href;
        return (
          <Link
            key={link.href}
            href={link.href}
            aria-current={active ? "page" : undefined}
            className={
              "focus-visible:ring-ring font-title shrink-0 rounded-full px-4 py-1.5 text-[15px] transition-colors focus-visible:ring-2 focus-visible:outline-none " +
              (active
                ? "bg-foreground text-background"
                : "text-secondary-foreground hover:text-foreground")
            }
          >
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
