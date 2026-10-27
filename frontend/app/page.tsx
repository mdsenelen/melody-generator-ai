import Link from "next/link";

import { buttonClass } from "../components/ui/button";

const COLAB_NOTEBOOK_URL =
  "https://colab.research.google.com/github/mdsenelen/melody-generator-ai/blob/main/backend/melody_generation_ORDERED_FINAL_(1).ipynb";

const PIPELINE_STEPS = [
  {
    n: "01",
    title: "Record or upload",
    description: "Play or upload a clip of audio straight from your browser.",
    href: "/analyse",
    linkLabel: "Start on the analyse page",
  },
  {
    n: "02",
    title: "Analyse",
    description: "We transcribe pitch, chords, key and mood from what you gave us.",
    href: "/analyse",
    linkLabel: "See the analyse page",
  },
  {
    n: "03",
    title: "Choose a progression",
    description: "Pick a chord progression to shape the melody that gets generated.",
    href: "/choose-progression",
    linkLabel: "Browse progressions",
  },
  {
    n: "04",
    title: "Generate & listen",
    description: "A trained CVAE + IDDM-PPO model generates variants. Preview, compare, download.",
    href: "/generate-variants",
    linkLabel: "Generate variants",
  },
] as const;

export default function LandingPage() {
  return (
    <div>
      <section className="border-border -mx-4 border-b px-6 py-16 sm:-mx-6 md:px-16 md:py-24 lg:-mx-10">
        <div className="max-w-2xl xl:max-w-none">
          {/* One line from xl up: the sentence measures ~28.9px per 1px of
              font-size in Fraunces Light, and xl leaves 1152px (1280 minus
              the 64px side padding), so 2.4375rem (39px, ~1126px) is the
              largest that fits; 3rem (~1385px) fits from 2xl (1408px). No
              nowrap, so a narrower viewport or a bigger root font wraps
              instead of scrolling. */}
          <h1 className="font-display text-foreground text-4xl leading-tight font-light md:text-5xl xl:text-[2.4375rem] 2xl:text-5xl">
            Turn your humming into a new melody, or analyse an audio clip
          </h1>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Link
              href="/analyse"
              className={buttonClass("primary", {
                className: "text-foreground bg-[#263b50] px-6 py-3 hover:bg-[#324b65]",
              })}
            >
              Start analysing audio
            </Link>
            <a
              href={COLAB_NOTEBOOK_URL}
              target="_blank"
              rel="noopener noreferrer"
              className={buttonClass("ghost")}
            >
              View the training notebook ↗
            </a>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-2xl py-14">
        <p className="text-muted-foreground mb-8 font-mono text-[10px] tracking-widest uppercase">
          How it works
        </p>
        <div className="space-y-0">
          {PIPELINE_STEPS.map((step, index) => (
            <div key={step.title} className="group flex gap-6">
              <div className="flex flex-col items-center">
                <div className="border-border group-hover:border-primary/40 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border transition-colors">
                  <span className="text-muted-foreground font-mono text-[9px]">{step.n}</span>
                </div>
                {index < PIPELINE_STEPS.length - 1 ? (
                  <div className="bg-border my-1 w-px flex-1" style={{ minHeight: 32 }} />
                ) : null}
              </div>
              <div className="pb-8">
                <p className="font-display text-foreground mb-1 text-base font-light">
                  {step.title}
                </p>
                <p className="text-muted-foreground text-sm leading-relaxed">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
          {[
            ["/analyse", "Analyse"],
            ["/listen-progressions", "Progressions"],
            ["/choose-progression", "Compose"],
            ["/generate-variants", "Variants"],
          ].map(([href, label]) => (
            <Link
              key={href}
              href={href}
              className="border-border text-muted-foreground hover:border-primary/40 hover:text-primary rounded-[var(--radius)] border py-2.5 text-center font-mono text-xs tracking-widest uppercase transition-all"
            >
              {label}
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
