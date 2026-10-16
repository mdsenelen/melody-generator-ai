const SENTENCE_PAUSE_S = 2;

type Word = { text: string; delay: number; duration: number };

// Each word is "written" left to right; longer words take longer, like a pen would.
function scheduleWords(text: string): Word[] {
  let start = 0.2;
  return text.split(" ").map((word) => {
    const duration = 0.06 + word.length * 0.02;
    const scheduled = { text: word, delay: start, duration };
    // Lift the pen for a beat at the end of each sentence.
    start += duration + (/[.!?]$/.test(word) ? SENTENCE_PAUSE_S : 0.015);
    return scheduled;
  });
}

/** A sentence that writes itself in by hand, word by word. Reduced motion shows it at once. */
export function HandwrittenIntro({ text }: { text: string }) {
  const words = scheduleWords(text);

  return (
    <p className="font-title text-foreground max-w-4xl text-[clamp(2.25rem,5vw,3.25rem)] leading-[1.2] text-balance">
      <span className="sr-only">{text}</span>
      <span aria-hidden="true">
        {words.map((word, index) => (
          <span key={`${word.text}-${index}`}>
            <span
              data-word
              className="ink-word"
              style={{
                animationDelay: `${word.delay.toFixed(2)}s`,
                animationDuration: `${word.duration.toFixed(2)}s`,
              }}
            >
              {word.text}
            </span>{" "}
          </span>
        ))}
      </span>
    </p>
  );
}
