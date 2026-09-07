import Link from "next/link";

const COLAB_NOTEBOOK_URL =
  "https://colab.research.google.com/github/mdsenelen/melody-generator-ai/blob/main/backend/melody_generation_ORDERED_FINAL_(1).ipynb";

const PIPELINE_STEPS = [
  {
    title: "Record or upload",
    description: "Play or upload a clip of audio straight from your browser.",
    href: "/analyse",
    linkLabel: "Start on the analyse page",
  },
  {
    title: "Analyse",
    description: "We transcribe pitch, chords, key and mood from what you gave us.",
    href: "/analyse",
    linkLabel: "See the analyse page",
  },
  {
    title: "Choose a progression",
    description: "Pick a chord progression to shape the melody that gets generated.",
    href: "/choose-progression",
    linkLabel: "Browse progressions",
  },
  {
    title: "Generate & listen",
    description: "A trained CVAE + IDDM-PPO model generates variants — preview, compare, download.",
    href: "/generate-variants",
    linkLabel: "Generate variants",
  },
] as const;

export default function LandingPage() {
  return (
    <main className="space-y-16 pb-8">
      <section className="px-2 pb-8 pt-6">
        <div className="mx-auto max-w-5xl">
          <div className="mb-8 inline-flex items-center rounded-[0.55rem] border border-[#8b5cf6]/75 bg-transparent px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.24em] text-[#d19af7]">
            AI music generation
          </div>

          <h1
            className="max-w-5xl text-[clamp(2.8rem,5vw,5.2rem)] leading-[0.94] tracking-[-0.07em] text-[#f3f6fb]"
            style={{ fontFamily: "var(--font-display)" }}
          >
            Turn a clip of audio into new,
            <span className="block text-[#edf3ff]">playable melodies</span>
          </h1>

          <p className="mt-8 max-w-3xl text-[1.08rem] leading-relaxed text-[#dfe7f5]/75 sm:text-[1.45rem]">
            Upload or record a short clip. Melodia analyses its harmonic content, then generates
            melodic variants you can preview, download, and use immediately — as MIDI or WAV.
          </p>

          <div className="mt-10 flex flex-wrap items-center gap-4">
            <Link
              href="/analyse"
              className="inline-flex items-center justify-center rounded-[0.9rem] border border-[#8b5cf6]/70 bg-[#8b5cf6] px-7 py-4 text-base font-semibold text-white shadow-[0_10px_30px_rgba(139,92,246,0.38)] transition hover:-translate-y-0.5 hover:shadow-[0_14px_36px_rgba(139,92,246,0.45)]"
            >
              Start analysing audio
            </Link>
            <a
              href={COLAB_NOTEBOOK_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center rounded-[0.9rem] border border-white/15 bg-white/5 px-7 py-4 text-base font-semibold text-white/85 transition hover:border-white/30 hover:bg-white/10"
            >
              View the training notebook ↗
            </a>
          </div>
        </div>
      </section>

      <section className="space-y-6">
        <h2 className="text-center text-sm font-semibold tracking-[0.2em] text-white/45 uppercase">
          How it works
        </h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {PIPELINE_STEPS.map((step, index) => (
            <Link
              key={step.title}
              href={step.href}
              className="group flex flex-col gap-3 rounded-[1.6rem] border border-white/10 bg-[rgba(17,22,32,0.7)] p-5 shadow-[0_6px_24px_rgba(2,6,23,0.2)] backdrop-blur-sm transition hover:border-[#8b5cf6]/45 hover:bg-[rgba(20,27,38,0.82)]"
            >
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-white/15 bg-[#1b2432] text-sm font-semibold text-[#f0e9ff] shadow-[inset_0_0_16px_rgba(139,92,246,0.12)]">
                {index + 1}
              </span>
              <p className="text-[1.05rem] font-semibold text-white">{step.title}</p>
              <p className="text-sm leading-6 text-white/65">{step.description}</p>
              <span className="mt-auto text-xs font-semibold text-[#d7b9ff] group-hover:text-[#f0d9ff]">
                {step.linkLabel} →
              </span>
            </Link>
          ))}
        </div>
      </section>

      <section className="rounded-[1.8rem] border border-white/10 bg-[rgba(17,22,32,0.7)] p-6 shadow-[0_14px_42px_rgba(2,6,23,0.25)] backdrop-blur-md">
        <h2 className="text-sm font-semibold tracking-[0.2em] text-white/45 uppercase">
          Good to know
        </h2>
        <ul className="mt-4 space-y-4 text-base leading-7 text-white/70">
          <li>
            The backend runs on Render&apos;s free tier — the first analysis after a period of
            inactivity can take a minute or two while it wakes up. Later ones are much faster.
          </li>
          <li>
            The CVAE + IDDM-PPO generation model is trained in the linked Colab notebook — open it
            to see how transcription, mood detection, and melody generation are trained end to end.
          </li>
          <li>
            No account needed — everything runs from your browser, and audio is only kept as long as
            it takes to generate your results.
          </li>
        </ul>
      </section>
    </main>
  );
}
