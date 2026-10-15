import Link from "next/link";

import { buttonClass } from "../components/ui/button";
import { HandwrittenIntro } from "../components/handwritten-intro";
import { Label } from "../components/ui/text";

const COLAB_NOTEBOOK_URL =
  "https://colab.research.google.com/github/mdsenelen/melody-generator-ai/blob/main/backend/melody_generation_ORDERED_FINAL_(1).ipynb";

const INTRO =
  "I am an assistant tool to help you make a song from a simple hum or a phrase. You can upload a song too, and I will generate alternative variants from that.";

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
        <div className="mx-auto flex min-h-[420px] max-w-4xl flex-col items-center justify-center gap-12">
          <h1 className="sr-only">Melodia</h1>
          <HandwrittenIntro text={INTRO} />

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
