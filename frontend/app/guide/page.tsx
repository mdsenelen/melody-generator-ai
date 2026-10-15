import Link from "next/link";

import { PageHeader } from "../../components/ui/text";

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
    </div>
  );
}
