"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type NavLink = { href: string; label: string; tip?: string; beta?: boolean };

const links: NavLink[] = [
  { href: "/guide", label: "User's guide" },
  { href: "/analyse", label: "Analyse", tip: "Take a look at the chords used and get a MIDI" },
  {
    href: "/listen-progressions",
    label: "Progressions",
    tip: "Create melodies from preset popular chord progressions",
  },
  { href: "/choose-progression", label: "Compose", tip: "Compose from scratch" },
  {
    href: "/generate-variants",
    label: "Variants",
    tip: "Generate alternative variants of your song",
    beta: true,
  },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Main"
      className="border-border bg-card/80 flex max-w-full items-center gap-1 overflow-x-auto rounded-full border p-1 shadow-[var(--shadow-panel)] md:overflow-visible"
    >
      {links.map((link) => {
        const active = pathname === link.href;
        const tipId = link.tip ? `nav-tip-${link.href.slice(1)}` : undefined;
        return (
          <div key={link.href} className="group relative shrink-0">
            <Link
              href={link.href}
              aria-current={active ? "page" : undefined}
              aria-describedby={tipId}
              className={
                "focus-visible:ring-ring font-title block rounded-full px-4 py-1.5 text-[15px] transition-colors focus-visible:ring-2 focus-visible:outline-none " +
                (active
                  ? "bg-foreground text-background"
                  : "text-secondary-foreground hover:text-foreground")
              }
            >
              {link.label}
            </Link>
            {link.tip ? (
              // Pops in under its link on hover or keyboard focus.
              <div
                id={tipId}
                role="tooltip"
                className="pointer-events-none invisible absolute top-full left-1/2 z-40 mt-3 w-56 -translate-x-1/2 -translate-y-1 scale-95 opacity-0 transition duration-150 ease-out group-focus-within:visible group-focus-within:translate-y-0 group-focus-within:scale-100 group-focus-within:opacity-100 group-hover:visible group-hover:translate-y-0 group-hover:scale-100 group-hover:opacity-100 motion-reduce:transform-none motion-reduce:transition-none"
              >
                <span
                  aria-hidden="true"
                  className="border-border-strong bg-raised absolute -top-1.5 left-1/2 h-3 w-3 -translate-x-1/2 rotate-45 border-t border-l"
                />
                <span className="border-border-strong bg-raised text-foreground relative block rounded-[var(--radius)] border px-3 py-2 text-center text-[13px] leading-5 shadow-[var(--shadow-panel)]">
                  {link.beta ? (
                    <span className="bg-beta text-beta-foreground mb-1.5 inline-block rounded-full px-2 py-0.5 font-mono text-[10px] font-medium tracking-[0.14em] uppercase">
                      Beta
                    </span>
                  ) : null}
                  <span className="block">{link.tip}</span>
                </span>
              </div>
            ) : null}
          </div>
        );
      })}
    </nav>
  );
}
