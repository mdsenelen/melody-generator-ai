import Link from "next/link";

import { buttonClass } from "../../components/ui/button";
import { PageHeader } from "../../components/ui/text";

const COLAB_NOTEBOOK_URL =
  "https://colab.research.google.com/github/mdsenelen/melody-generator-ai/blob/main/backend/melody_generation_ORDERED_FINAL_(1).ipynb";

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
    body: "No account needed, everything runs from your browser, and audio is only kept as long as it takes to generate your results.",
  },
];

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

export default function GuidePage() {
  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="User's guide"
        title="How it works"
        lead="From a hum to a melody, in four steps."
      />
      <ol className="grid gap-4 md:grid-cols-2">
        {PIPELINE_STEPS.map((step) => (
          <li
            key={step.title}
            className="border-border bg-card hover:border-border-strong flex flex-col rounded-[var(--radius-lg)] border p-7 transition-colors"
          >
            <span className="text-primary-bright font-mono text-[13px] font-medium">{step.n}</span>
            <p className="font-title text-foreground mt-5 text-2xl">{step.title}</p>
            <p className="text-secondary-foreground mt-2 flex-1 text-[15px] leading-6">
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
      <section aria-labelledby="good-to-know" className="border-border border-t pt-10">
        <h2
          id="good-to-know"
          className="text-muted-foreground font-serif text-[26px] leading-[1.2] font-normal uppercase"
        >
          Good to know
        </h2>
        <dl className="mt-6 grid gap-8 md:grid-cols-3">
          {GOOD_TO_KNOW.map((item) => (
            <div key={item.title}>
              <dt className="font-title text-foreground text-lg">{item.title}</dt>
              <dd className="text-secondary-foreground mt-2 font-serif text-[26px] leading-[1.2]">
                {item.body}
              </dd>
            </div>
          ))}
        </dl>
      </section>
      <div className="font-title flex flex-wrap items-center justify-center gap-3">
        <Link href="/analyse" className={buttonClass("primary", { className: "font-title" })}>
          Start analysing audio <span aria-hidden="true">↗</span>
        </Link>
        <a
          href={COLAB_NOTEBOOK_URL}
          target="_blank"
          rel="noopener noreferrer"
          className={buttonClass("ghost", { className: "font-title" })}
        >
          View the training notebook ↗
        </a>
      </div>
    </div>
  );
}
