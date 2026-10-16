import { HandwrittenIntro } from "../components/handwritten-intro";

const INTRO =
  "I am an assistant tool to help you make a song from a simple hum or a phrase. You can upload a song too, and I will generate alternative variants from that.";

export default function LandingPage() {
  return (
    <div className="space-y-24">
      <section className="relative isolate pt-10 pb-6 text-center md:pt-16">
        <div className="mx-auto flex min-h-[420px] max-w-4xl flex-col items-center justify-center gap-12">
          <h1 className="sr-only">Melodia</h1>
          <HandwrittenIntro text={INTRO} />
        </div>
      </section>
    </div>
  );
}
