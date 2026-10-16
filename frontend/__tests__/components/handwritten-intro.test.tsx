import { render, screen } from "@testing-library/react";

import { HandwrittenIntro } from "../../components/handwritten-intro";

const TEXT = "I can turn a hum into a song.";

describe("HandwrittenIntro", () => {
  it("gives screen readers the sentence once, not word by word", () => {
    const { container } = render(<HandwrittenIntro text={TEXT} />);
    expect(screen.getByText(TEXT)).toBeInTheDocument();
    const animated = container.querySelector("[aria-hidden='true']");
    expect(animated?.querySelectorAll("[data-word]")).toHaveLength(8);
  });

  it("writes the words in order, each starting after the one before", () => {
    const { container } = render(<HandwrittenIntro text={TEXT} />);
    const starts = Array.from(container.querySelectorAll<HTMLElement>("[data-word]")).map((word) =>
      parseFloat(word.style.animationDelay),
    );
    expect(starts).toEqual([...starts].sort((a, b) => a - b));
    expect(new Set(starts).size).toBe(starts.length);
  });

  it("pauses two seconds after a sentence ends before writing the next one", () => {
    const { container } = render(<HandwrittenIntro text="Hum a tune. Hear it back." />);
    const words = Array.from(container.querySelectorAll<HTMLElement>("[data-word]"));
    const start = (i: number) => parseFloat(words[i].style.animationDelay);
    const end = (i: number) => start(i) + parseFloat(words[i].style.animationDuration);
    // "tune." is word 2; "Hear" (word 3) waits at least 2s after it finishes.
    expect(start(3) - end(2)).toBeGreaterThanOrEqual(2);
    // Words inside a sentence follow on quickly.
    expect(start(1) - end(0)).toBeLessThan(0.5);
  });
});
