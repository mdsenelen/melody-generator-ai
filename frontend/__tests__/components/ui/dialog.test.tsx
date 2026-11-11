import { createRef, useState } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Button } from "../../../components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogTrigger,
} from "../../../components/ui/dialog";

function Example(props: {
  description?: string;
  tone?: "default" | "destructive";
  onOpenChange?: (o: boolean) => void;
  open?: boolean;
  defaultOpen?: boolean;
}) {
  const { description, tone, ...root } = props;
  return (
    <Dialog {...root}>
      <DialogTrigger asChild>
        <Button>Delete take</Button>
      </DialogTrigger>
      <DialogContent title="Delete this take?" description={description} tone={tone}>
        <input aria-label="Reason" />
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="ghost">Keep take</Button>
          </DialogClose>
          <Button variant="destructive">Delete take now</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

describe("Dialog", () => {
  it("is closed by default and the trigger opens it", async () => {
    const user = userEvent.setup();
    render(<Example />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Delete take" }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("is named by its title and described when a description is given", async () => {
    const user = userEvent.setup();
    render(<Example description="This cannot be undone." />);
    await user.click(screen.getByRole("button", { name: "Delete take" }));
    const dialog = screen.getByRole("dialog", { name: "Delete this take?" });
    expect(dialog).toHaveAccessibleDescription("This cannot be undone.");
  });

  it("has no description relationship when none is given, and no Radix warning", async () => {
    const user = userEvent.setup();
    const warn = jest.spyOn(console, "warn").mockImplementation(() => {});
    const error = jest.spyOn(console, "error").mockImplementation(() => {});
    render(<Example />);
    await user.click(screen.getByRole("button", { name: "Delete take" }));
    expect(screen.getByRole("dialog")).not.toHaveAccessibleDescription();
    expect(warn).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
    warn.mockRestore();
    error.mockRestore();
  });

  it("requires a title at the type level", () => {
    // @ts-expect-error DialogContent without `title` does not compile
    const untitled = <DialogContent>x</DialogContent>;
    expect(untitled).toBeTruthy();
  });

  it("moves focus inside on open and cycles Tab within the dialog", async () => {
    const user = userEvent.setup();
    render(<Example />);
    await user.click(screen.getByRole("button", { name: "Delete take" }));
    const dialog = screen.getByRole("dialog");
    await waitFor(() => expect(dialog).toContainElement(document.activeElement as HTMLElement));
    for (let i = 0; i < 8; i += 1) {
      await user.tab();
      expect(dialog).toContainElement(document.activeElement as HTMLElement);
    }
    await user.tab({ shift: true });
    expect(dialog).toContainElement(document.activeElement as HTMLElement);
  });

  it("makes the background inert to assistive tech while open", async () => {
    const user = userEvent.setup();
    render(<Example />);
    await user.click(screen.getByRole("button", { name: "Delete take" }));
    expect(screen.queryByRole("button", { name: "Delete take" })).not.toBeInTheDocument();
  });

  it("Escape closes and focus returns to the trigger", async () => {
    const user = userEvent.setup();
    render(<Example />);
    const trigger = screen.getByRole("button", { name: "Delete take" });
    await user.click(trigger);
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(trigger).toHaveFocus();
  });

  it("has a named close button that closes", async () => {
    const user = userEvent.setup();
    render(<Example />);
    await user.click(screen.getByRole("button", { name: "Delete take" }));
    await user.click(screen.getByRole("button", { name: "Close" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("DialogClose in the footer closes too", async () => {
    const user = userEvent.setup();
    render(<Example />);
    await user.click(screen.getByRole("button", { name: "Delete take" }));
    await user.click(screen.getByRole("button", { name: "Keep take" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("supports controlled open / onOpenChange", async () => {
    const user = userEvent.setup();
    const onOpenChange = jest.fn();
    const { rerender } = render(<Example open={false} onOpenChange={onOpenChange} />);
    await user.click(screen.getByRole("button", { name: "Delete take" }));
    expect(onOpenChange).toHaveBeenCalledWith(true);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    rerender(<Example open onOpenChange={onOpenChange} />);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(onOpenChange).toHaveBeenLastCalledWith(false);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("works with state owned by the caller", async () => {
    const user = userEvent.setup();
    function Owner() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button onClick={() => setOpen(true)}>Open it</button>
          <Dialog open={open} onOpenChange={setOpen}>
            <DialogContent title="Owned">body</DialogContent>
          </Dialog>
        </>
      );
    }
    render(<Owner />);
    await user.click(screen.getByRole("button", { name: "Open it" }));
    expect(screen.getByRole("dialog", { name: "Owned" })).toBeInTheDocument();
  });

  it("exposes the destructive tone as data, and className/ref reach the content", async () => {
    const ref = createRef<HTMLDivElement>();
    render(
      <Dialog defaultOpen>
        <DialogContent ref={ref} title="T" tone="destructive" className="custom-content max-w-xs">
          x
        </DialogContent>
      </Dialog>,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("data-tone", "destructive");
    expect(dialog).toHaveClass("custom-content", "max-w-xs");
    expect(dialog).not.toHaveClass("max-w-md");
    expect(ref.current).toBe(dialog);
  });

  it("a busy close control: loading blocks closing via the close button", async () => {
    const user = userEvent.setup();
    const onOpenChange = jest.fn();
    render(
      <Dialog defaultOpen onOpenChange={onOpenChange}>
        <DialogContent title="Working" loading>
          x
        </DialogContent>
      </Dialog>,
    );
    expect(screen.getByRole("dialog")).toHaveAttribute("aria-busy", "true");
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(onOpenChange).not.toHaveBeenCalled();
  });
});
