import Link from "next/link";

import { buttonClass } from "../components/ui/button";
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
    title: "Cold-server latency",
    body: "The backend runs on Render's free tier — the first analysis after a period of inactivity can take a minute or two while it wakes up. Later ones are much faster.",
  },
  {
    title: "Trained end to end",
    body: "The CVAE + IDDM-PPO generation model is trained in the linked Colab notebook — open it to see how transcription, mood detection, and melody generation are trained.",
  },
  {
    title: "Privacy",
    body: "No account needed — everything runs from your browser, and audio is only kept as long as it takes to generate your results.",
  },
];

export default function LandingPage() {
  return (
    <div className="space-y-24">
      <section className="relative isolate pt-10 pb-6 text-center md:pt-16">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 -top-24 -z-10 mx-auto h-[420px] max-w-3xl rounded-full bg-[radial-gradient(closest-side,rgba(143,169,204,0.16),transparent)] blur-2xl"
        />
        <div className="mx-auto max-w-4xl">
          <h1 className="sr-only">Melodia</h1>

          <div className="flex flex-wrap items-center justify-center gap-3">
            <Link href="/analyse" className={buttonClass("primary")}>
              Start analysing audio <span aria-hidden="true">↗</span>
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

      <section aria-labelledby="how-it-works">
        <div className="mb-8 flex items-end justify-between gap-4">
          <div>
            <Label>How it works</Label>
            <h2 id="how-it-works" className="font-title text-foreground mt-3 text-3xl">
              From a hum to a melody, in four steps
            </h2>
          </div>
        </div>
        <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {PIPELINE_STEPS.map((step) => (
            <li
              key={step.title}
              className="border-border bg-card hover:border-border-strong flex flex-col rounded-[var(--radius-lg)] border p-6 transition-colors"
            >
              <span className="text-primary-bright font-mono text-[13px] font-medium">
                {step.n}
              </span>
              <p className="font-title text-foreground mt-6 text-xl">{step.title}</p>
              <p className="text-secondary-foreground mt-2 flex-1 text-sm leading-6">
                {step.description}
              </p>
              <Link
                href={step.href}
                className="text-primary-bright focus-visible:ring-ring mt-5 inline-block self-start rounded-sm text-sm hover:underline focus-visible:ring-2 focus-visible:outline-none"
              >
                {step.linkLabel} →
              </Link>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="good-to-know" className="border-border border-t pt-12">
        <Label>Good to know</Label>
        <h2 id="good-to-know" className="sr-only">
          Good to know
        </h2>
        <dl className="mt-6 grid gap-8 md:grid-cols-3">
          {GOOD_TO_KNOW.map((item) => (
            <div key={item.title}>
              <dt className="font-title text-foreground text-lg">{item.title}</dt>
              <dd className="text-secondary-foreground mt-2 text-sm leading-6">{item.body}</dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
  );
}
