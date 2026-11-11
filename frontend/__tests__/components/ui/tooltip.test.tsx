import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { IconButton } from "../../../components/ui/icon-button";
import { Tooltip, TooltipProvider, TOOLTIP_DELAY_MS } from "../../../components/ui/tooltip";

describe("Tooltip", () => {
  it("works as a lone component with no provider and is hidden by default", () => {
    render(
      <Tooltip content="Play from the start">
        <button>Play</button>
      </Tooltip>,
    );
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("shows on hover and links the trigger by aria-describedby", async () => {
    const user = userEvent.setup();
    render(
      <Tooltip content="Play from the start" delayDuration={0}>
        <button>Play</button>
      </Tooltip>,
    );
    const trigger = screen.getByRole("button", { name: "Play" });
    await user.hover(trigger);
    const tip = await screen.findByRole("tooltip");
    expect(tip).toHaveTextContent("Play from the start");
    expect(trigger.getAttribute("aria-describedby")).toBe(tip.id);
  });

  it("shows on keyboard focus", async () => {
    const user = userEvent.setup();
    render(
      <Tooltip content="Play from the start" delayDuration={0}>
        <button>Play</button>
      </Tooltip>,
    );
    await user.tab();
    expect(await screen.findByRole("tooltip")).toBeInTheDocument();
  });

  it("hides on Escape", async () => {
    const user = userEvent.setup();
    render(
      <Tooltip content="Play from the start" delayDuration={0}>
        <button>Play</button>
      </Tooltip>,
    );
    await user.tab();
    await screen.findByRole("tooltip");
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("tooltip")).not.toBeInTheDocument());
  });

  it("supplements, never replaces, the trigger's accessible name", async () => {
    const user = userEvent.setup();
    render(
      <Tooltip content="Space" delayDuration={0}>
        <IconButton label="Play" icon={<svg viewBox="0 0 8 8" />} />
      </Tooltip>,
    );
    await user.tab();
    await screen.findByRole("tooltip");
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
  });

  it("does not open when disabled by the caller", async () => {
    const user = userEvent.setup();
    render(
      <Tooltip content="Nope" delayDuration={0} open={false}>
        <button>Play</button>
      </Tooltip>,
    );
    await user.hover(screen.getByRole("button"));
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("shares one default delay constant, and nests inside an app-level provider", async () => {
    expect(TOOLTIP_DELAY_MS).toBeGreaterThan(0);
    const user = userEvent.setup();
    render(
      <TooltipProvider>
        <Tooltip content="Grouped" delayDuration={0}>
          <button>Play</button>
        </Tooltip>
      </TooltipProvider>,
    );
    await user.tab();
    expect(await screen.findByRole("tooltip")).toHaveTextContent("Grouped");
  });

  it("content is a string, so it cannot hold interactive content", () => {
    // @ts-expect-error tooltip content must be plain text, not elements
    const bad = <Tooltip content={<button>x</button>}>t</Tooltip>;
    expect(bad).toBeTruthy();
  });
});
