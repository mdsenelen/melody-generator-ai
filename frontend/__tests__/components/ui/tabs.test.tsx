import { createRef, useState } from "react";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../../components/ui/tabs";

function Basic(props: Partial<React.ComponentProps<typeof Tabs>> & { listClass?: string }) {
  return (
    <Tabs defaultValue="notes" {...props}>
      <TabsList aria-label="Analysis views" className={props.listClass}>
        <TabsTrigger value="notes">Notes</TabsTrigger>
        <TabsTrigger value="chords" disabled>
          Chords
        </TabsTrigger>
        <TabsTrigger value="key">Key</TabsTrigger>
      </TabsList>
      <TabsContent value="notes">Notes panel</TabsContent>
      <TabsContent value="chords">Chords panel</TabsContent>
      <TabsContent value="key">Key panel</TabsContent>
    </Tabs>
  );
}

describe("Tabs", () => {
  it("exposes tablist, tab and tabpanel roles, the list named by aria-label", () => {
    render(<Basic />);
    expect(screen.getByRole("tablist", { name: "Analysis views" })).toBeInTheDocument();
    expect(screen.getAllByRole("tab")).toHaveLength(3);
    expect(screen.getByRole("tabpanel")).toHaveTextContent("Notes panel");
  });

  it("accepts aria-labelledby as the list name", () => {
    render(
      <>
        <h2 id="h">Views</h2>
        <Tabs defaultValue="a">
          <TabsList aria-labelledby="h">
            <TabsTrigger value="a">A</TabsTrigger>
          </TabsList>
          <TabsContent value="a">a</TabsContent>
        </Tabs>
      </>,
    );
    expect(screen.getByRole("tablist", { name: "Views" })).toBeInTheDocument();
  });

  it("requires a name on TabsList at the type level", () => {
    // @ts-expect-error a TabsList with neither aria-label nor aria-labelledby does not compile
    const unnamed = <TabsList>x</TabsList>;
    expect(unnamed).toBeTruthy();
  });

  it("click selects a tab and shows only its panel", async () => {
    const user = userEvent.setup();
    render(<Basic />);
    await user.click(screen.getByRole("tab", { name: "Key" }));
    expect(screen.getByRole("tab", { name: "Key" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Notes" })).toHaveAttribute("aria-selected", "false");
    expect(screen.getByText("Key panel")).toBeVisible();
    expect(screen.queryByText("Notes panel")).not.toBeInTheDocument();
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
  });

  it("wires aria-controls to the panel and aria-labelledby back to the tab", () => {
    render(<Basic />);
    const tab = screen.getByRole("tab", { name: "Notes" });
    const panel = screen.getByRole("tabpanel");
    expect(tab).toHaveAttribute("aria-controls", panel.id);
    expect(panel).toHaveAttribute("aria-labelledby", tab.id);
  });

  it("moves with arrow keys, skipping the disabled tab, and wraps", async () => {
    const user = userEvent.setup();
    render(<Basic />);
    act(() => screen.getByRole("tab", { name: "Notes" }).focus());
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Key" })).toHaveFocus();
    expect(screen.getByRole("tab", { name: "Key" })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Notes" })).toHaveFocus();
    await user.keyboard("{ArrowLeft}");
    expect(screen.getByRole("tab", { name: "Key" })).toHaveFocus();
  });

  it("Home and End jump to the first and last enabled tab", async () => {
    const user = userEvent.setup();
    render(<Basic defaultValue="key" />);
    act(() => screen.getByRole("tab", { name: "Key" }).focus());
    await user.keyboard("{Home}");
    expect(screen.getByRole("tab", { name: "Notes" })).toHaveFocus();
    expect(screen.getByRole("tab", { name: "Notes" })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{End}");
    expect(screen.getByRole("tab", { name: "Key" })).toHaveFocus();
  });

  it("a disabled tab cannot be activated", async () => {
    const user = userEvent.setup();
    render(<Basic />);
    const disabled = screen.getByRole("tab", { name: "Chords" });
    expect(disabled).toBeDisabled();
    await user.click(disabled);
    expect(disabled).toHaveAttribute("aria-selected", "false");
    expect(screen.getByText("Notes panel")).toBeInTheDocument();
  });

  it("works uncontrolled and controlled", async () => {
    const user = userEvent.setup();
    const onValueChange = jest.fn();
    function Controlled() {
      const [value, setValue] = useState("notes");
      return (
        <Basic
          value={value}
          onValueChange={(v) => {
            onValueChange(v);
            setValue(v);
          }}
        />
      );
    }
    render(<Controlled />);
    await user.click(screen.getByRole("tab", { name: "Key" }));
    expect(onValueChange).toHaveBeenCalledWith("key");
    expect(screen.getByText("Key panel")).toBeInTheDocument();
  });

  it("a controlled value that never changes keeps the tab selected", async () => {
    const user = userEvent.setup();
    render(<Basic value="notes" onValueChange={() => {}} />);
    await user.click(screen.getByRole("tab", { name: "Key" }));
    expect(screen.getByRole("tab", { name: "Notes" })).toHaveAttribute("aria-selected", "true");
  });

  it("marks a loading panel busy and keeps it reachable", () => {
    render(
      <Tabs defaultValue="a">
        <TabsList aria-label="L">
          <TabsTrigger value="a" loading>
            A
          </TabsTrigger>
        </TabsList>
        <TabsContent value="a" loading>
          a
        </TabsContent>
      </Tabs>,
    );
    expect(screen.getByRole("tab", { name: /A/ })).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("tabpanel")).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("tab", { name: /A/ })).not.toBeDisabled();
  });

  it("makes an empty panel focusable so keyboard users can reach it", () => {
    render(<Basic />);
    expect(screen.getByRole("tabpanel")).toHaveAttribute("tabindex", "0");
  });

  it("className overrides win and refs forward", () => {
    const ref = createRef<HTMLButtonElement>();
    render(
      <Tabs defaultValue="a">
        <TabsList aria-label="L" className="custom-list px-9">
          <TabsTrigger ref={ref} value="a" className="custom-trigger">
            A
          </TabsTrigger>
        </TabsList>
        <TabsContent value="a" className="custom-panel">
          a
        </TabsContent>
      </Tabs>,
    );
    expect(screen.getByRole("tablist")).toHaveClass("custom-list", "px-9");
    expect(screen.getByRole("tablist")).not.toHaveClass("px-1");
    expect(ref.current).toHaveClass("custom-trigger");
    expect(screen.getByRole("tabpanel")).toHaveClass("custom-panel");
  });
});
