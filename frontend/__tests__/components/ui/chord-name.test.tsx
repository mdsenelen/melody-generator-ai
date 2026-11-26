import { render, screen } from "@testing-library/react";

import { ChordName } from "../../../components/ui/chord-name";
import { ChordGraph } from "../../../components/chord-graph";
import { ChordDiagram } from "../../../components/chord-diagram";
import { renderWithQueryClient } from "../../../src/test/query/render";

describe("ChordName", () => {
  it("writes the usual name plainly and the other spelling in italics", () => {
    const { container } = render(<ChordName chord="D#m7" />);
    expect(container.textContent).toBe("Ebm7 / D#m7");
    // only the alternate is inside <i>; the usual name is not
    expect(container.querySelector("i")?.textContent).toBe("D#m7");
  });

  it("shows only one name when there is no enharmonic twin", () => {
    const { container } = render(<ChordName chord="Am7" />);
    expect(container.textContent).toBe("Am7");
    expect(container.querySelector("i")).toBeNull();
  });

  it("applies a label formatter to both names", () => {
    const { container } = render(<ChordName chord="Eb" format={(name) => `<${name}>`} />);
    expect(container.textContent).toBe("<Eb> / <D#>");
    expect(container.querySelector("i")?.textContent).toBe("<D#>");
  });
});

describe("chord names where chords are listed", () => {
  it("ChordDiagram button shows the italic alternate", () => {
    render(<ChordDiagram chord="D#" />);
    const button = screen.getByRole("button", { name: /play eb/i });
    expect(button.querySelector("i")?.textContent).toMatch(/D#/);
  });

  it("ChordGraph's sequence chips show the italic alternate", () => {
    renderWithQueryClient(<ChordGraph title="Test" progression={["C", "G#", "Fm"]} />);
    // the "Chord sequence" chip and the diagram button both carry the italic twin
    expect(screen.getAllByText("G#", { selector: "i" }).length).toBeGreaterThanOrEqual(2);
  });
});
