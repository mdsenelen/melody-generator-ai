import Link from "next/link";

import { buttonClass } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Label } from "../components/ui/text";

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
    description: "A trained CVAE + IDDM-PPO model generates variants — preview, compare, download.",
    href: "/generate-variants",
    linkLabel: "Generate variants",
  },
] as const;

const GOOD_TO_KNOW = [
  {
    icon: "⏱",
    title: "Cold-server latency",
    body: "The backend runs on Render's free tier — the first analysis after a period of inactivity can take a minute or two while it wakes up. Later ones are much faster.",
  },
  {
    icon: "🎵",
    title: "Trained end to end",
    body: "The CVAE + IDDM-PPO generation model is trained in the linked Colab notebook — open it to see how transcription, mood detection, and melody generation are trained.",
  },
  {
    icon: "🔒",
    title: "Privacy",
    body: "No account needed — everything runs from your browser, and audio is only kept as long as it takes to generate your results.",
  },
];

export default function LandingPage() {
  return (
    <div className="space-y-16">
      <section className="pt-4 pb-8 text-center">
        <div className="mx-auto max-w-3xl">
          <div className="border-primary/75 text-primary mb-8 inline-flex items-center rounded-[var(--radius)] border px-3 py-1.5 font-mono text-[10px] tracking-widest uppercase">
            AI music generation
          </div>

          <h1 className="font-display text-foreground text-[clamp(2.4rem,5vw,4.4rem)] leading-[0.98] font-light">
            Turn a clip of audio into new,
            <span className="block">playable melodies</span>
          </h1>

          <p className="text-secondary-foreground mx-auto mt-8 max-w-2xl text-base leading-relaxed">
            Upload or record a short clip. Melodia analyses its harmonic content, then generates
            melodic variants you can preview, download, and use immediately, as MIDI or WAV.
          </p>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <Link href="/analyse" className={buttonClass("primary")}>
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

      <section>
        <p className="text-muted-foreground mb-8 text-center font-mono text-[11px] tracking-widest uppercase">
          How it works
        </p>
        <div className="mx-auto max-w-2xl space-y-0">
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
                <Link
                  href={step.href}
                  className="text-primary mt-1 inline-block text-xs hover:underline"
                >
                  {step.linkLabel} →
                </Link>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-2xl">
        <Card className="space-y-4">
          <Label>Good to know</Label>
          <div className="space-y-4">
            {GOOD_TO_KNOW.map((item) => (
              <div key={item.title} className="space-y-1">
                <p className="text-foreground flex items-center gap-2 text-sm">
                  <span aria-hidden="true">{item.icon}</span>
                  {item.title}
                </p>
                <p className="text-muted-foreground pl-6 text-xs leading-relaxed">{item.body}</p>
              </div>
            ))}
          </div>
        </Card>
      </section>
    </div>
  );
}
