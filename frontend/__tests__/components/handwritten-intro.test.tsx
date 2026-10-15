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
});
