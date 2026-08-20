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
      <section className="flex flex-col items-center gap-6 rounded-[2rem] border border-white/10 bg-white/5 px-6 py-16 text-center shadow-xl shadow-black/20 backdrop-blur-md">
        <span className="rounded-full border border-purple-400/40 bg-purple-600/20 px-4 py-1 text-xs font-semibold tracking-[0.2em] text-purple-200 uppercase">
          AI music generation
        </span>
        <h1 className="max-w-2xl text-4xl font-semibold text-white sm:text-5xl">
          Turn a clip of audio into new, playable melodies
        </h1>
        <p className="max-w-xl text-base text-white/70 sm:text-lg">
          Upload or record audio, and MelodyAI transcribes the pitch, chords, key and mood, then
          generates new melody variants and chord progressions you can preview and download as MIDI
          or WAV.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/analyse"
            className="inline-flex items-center justify-center rounded-2xl border border-purple-400/40 bg-purple-600/20 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-purple-950/20 transition hover:border-purple-300 hover:bg-purple-500/25"
          >
            Start analysing audio
          </Link>
          <a
            href={COLAB_NOTEBOOK_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center rounded-2xl border border-white/15 bg-white/5 px-6 py-3 text-sm font-semibold text-white/85 transition hover:border-white/30 hover:bg-white/10"
          >
            View the training notebook ↗
          </a>
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
              className="group flex flex-col gap-3 rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm transition hover:border-purple-400/40 hover:bg-white/10"
            >
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-t from-purple-500 to-sky-400 text-sm font-semibold text-white">
                {index + 1}
              </span>
              <p className="text-sm font-semibold text-white">{step.title}</p>
              <p className="text-sm text-white/65">{step.description}</p>
              <span className="mt-auto text-xs font-semibold text-purple-300 group-hover:text-purple-200">
                {step.linkLabel} →
              </span>
            </Link>
          ))}
        </div>
      </section>

      <section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
        <h2 className="text-sm font-semibold tracking-[0.2em] text-white/45 uppercase">
          Good to know
        </h2>
        <ul className="mt-4 space-y-3 text-sm text-white/70">
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
