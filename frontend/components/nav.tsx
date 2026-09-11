"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/analyse", label: "Analyse" },
  { href: "/listen-progressions", label: "Progressions" },
  { href: "/choose-progression", label: "Compose" },
  { href: "/generate-variants", label: "Variants" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav className="flex flex-wrap items-center gap-5 font-mono text-[11px] tracking-widest uppercase">
      {links.map((link) => {
        const active = pathname === link.href;
        return (
          <Link
            key={link.href}
            href={link.href}
            aria-current={active ? "page" : undefined}
            className={
              active
                ? "text-primary transition-colors"
                : "text-muted-foreground hover:text-secondary-foreground transition-colors"
            }
          >
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
