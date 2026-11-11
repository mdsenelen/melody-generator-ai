import { createRef } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import {
  Button,
  buttonClass,
  type ButtonSize,
  type ButtonVariant,
} from "../../../components/ui/button";

describe("Button", () => {
  it("renders a real, enabled button with its accessible name", () => {
    render(<Button>Generate melody</Button>);
    const button = screen.getByRole("button", { name: "Generate melody" });
    expect(button.tagName).toBe("BUTTON");
    expect(button).toBeEnabled();
  });

  it("defaults to type=button and honours an explicit type", () => {
    const { rerender } = render(<Button>Go</Button>);
    expect(screen.getByRole("button")).toHaveAttribute("type", "button");
    rerender(<Button type="submit">Go</Button>);
    expect(screen.getByRole("button")).toHaveAttribute("type", "submit");
  });

  it("fires onClick on click, Enter and Space", async () => {
    const user = userEvent.setup();
    const onClick = jest.fn();
    render(<Button onClick={onClick}>Go</Button>);
    const button = screen.getByRole("button");
    await user.click(button);
    button.focus();
    await user.keyboard("{Enter}");
    await user.keyboard(" ");
    expect(onClick).toHaveBeenCalledTimes(3);
  });

  it("disabled uses the real attribute and blocks activation", async () => {
    const user = userEvent.setup();
    const onClick = jest.fn();
    render(
      <Button disabled onClick={onClick}>
        Go
      </Button>,
    );
    const button = screen.getByRole("button");
    expect(button).toBeDisabled();
    await user.click(button);
    expect(onClick).not.toHaveBeenCalled();
  });

  describe("loading", () => {
    it("sets aria-busy, keeps its name, and stays focusable", async () => {
      const user = userEvent.setup();
      render(<Button loading>Generate melody</Button>);
      const button = screen.getByRole("button", { name: "Generate melody" });
      expect(button).toHaveAttribute("aria-busy", "true");
      expect(button).toBeEnabled();
      await user.tab();
      expect(button).toHaveFocus();
    });

    it("blocks click, Enter and Space", async () => {
      const user = userEvent.setup();
      const onClick = jest.fn();
      render(
        <Button loading onClick={onClick}>
          Go
        </Button>,
      );
      const button = screen.getByRole("button");
      await user.click(button);
      button.focus();
      await user.keyboard("{Enter}");
      await user.keyboard(" ");
      expect(onClick).not.toHaveBeenCalled();
    });

    it("does not submit a form while loading", async () => {
      const user = userEvent.setup();
      const onSubmit = jest.fn((e: React.FormEvent) => e.preventDefault());
      render(
        <form onSubmit={onSubmit}>
          <Button type="submit" loading>
            Go
          </Button>
        </form>,
      );
      await user.click(screen.getByRole("button"));
      expect(onSubmit).not.toHaveBeenCalled();
    });

    it("keeps the label in the DOM (width stable) and marks the spinner aria-hidden", () => {
      render(<Button loading>Generate melody</Button>);
      expect(screen.getByText("Generate melody")).toBeInTheDocument();
      const spinner = screen.getByRole("button").querySelector("[data-button-spinner]");
      expect(spinner).not.toBeNull();
      expect(spinner).toHaveAttribute("aria-hidden", "true");
    });

    it("is not busy by default", () => {
      render(<Button>Go</Button>);
      expect(screen.getByRole("button")).not.toHaveAttribute("aria-busy");
    });
  });

  it("error: aria-invalid is passed through and adds a non-colour marker", () => {
    render(<Button aria-invalid="true">Retry</Button>);
    const button = screen.getByRole("button", { name: "Retry" });
    expect(button).toHaveAttribute("aria-invalid", "true");
    const marker = button.querySelector("[data-button-error]");
    expect(marker).not.toBeNull();
    expect(marker).toHaveAttribute("aria-hidden", "true");
  });

  it("has no error marker when valid", () => {
    render(<Button>Retry</Button>);
    expect(screen.getByRole("button").querySelector("[data-button-error]")).toBeNull();
  });

  describe("variants and sizes", () => {
    const variants: ButtonVariant[] = ["primary", "secondary", "ghost", "destructive"];
    const sizes: ButtonSize[] = ["sm", "md"];

    it.each(variants)("applies the %s variant distinctly", (variant) => {
      const classes = variants.map((v) => buttonClass(v));
      render(<Button variant={variant}>x</Button>);
      const el = screen.getByRole("button");
      expect(el.className).toContain(buttonClass(variant).split(" ").slice(-1)[0]);
      expect(new Set(classes).size).toBe(variants.length);
    });

    it("primary is dark-on-accent; destructive uses danger tokens", () => {
      expect(buttonClass("primary")).toContain("bg-accent");
      expect(buttonClass("primary")).toContain("text-accent-foreground");
      expect(buttonClass("destructive")).toContain("text-danger");
    });

    it.each(sizes)("applies the %s size", (size) => {
      render(<Button size={size}>x</Button>);
      expect(screen.getByRole("button").className).toContain(size === "sm" ? "text-xs" : "text-sm");
    });

    it("keeps the legacy `small` prop working as size sm", () => {
      render(<Button small>x</Button>);
      expect(screen.getByRole("button").className).toContain("text-xs");
    });

    it("uses the shared focus ring and tokens only", () => {
      render(<Button>x</Button>);
      const cls = screen.getByRole("button").className;
      expect(cls).toContain("focus-visible:outline-focus-ring");
      expect(cls).not.toMatch(/#[0-9a-f]{3,8}|\b(red|green|amber|sky)-\d{2,3}\b|ring-ring/i);
    });
  });

  it("className overrides built-in classes via cn()", () => {
    render(<Button className="w-full px-6">x</Button>);
    const cls = screen.getByRole("button").className.split(" ");
    expect(cls).toContain("px-6");
    expect(cls).toContain("w-full");
    expect(cls).not.toContain("px-4");
  });

  it("buttonClass keeps its legacy signature", () => {
    expect(buttonClass("secondary", { small: true, className: "mt-2" })).toContain("mt-2");
    expect(buttonClass("secondary", { small: true })).toContain("text-xs");
    expect(buttonClass()).toContain("bg-accent");
  });

  it("forwards its ref to the button element", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<Button ref={ref}>x</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });

  describe("asChild", () => {
    it("renders the child element with button styling and no nested button", () => {
      render(
        <Button asChild variant="secondary">
          <a href="/result/1">View result</a>
        </Button>,
      );
      const link = screen.getByRole("link", { name: "View result" });
      expect(link.tagName).toBe("A");
      expect(link).toHaveAttribute("href", "/result/1");
      expect(link.className).toContain("rounded");
      expect(link.className).toContain("focus-visible:outline-focus-ring");
      expect(link).not.toHaveAttribute("type");
      expect(document.querySelector("button")).toBeNull();
    });

    it("blocks navigation while loading and keeps the name", async () => {
      const user = userEvent.setup();
      const onClick = jest.fn();
      render(
        <Button asChild loading onClick={onClick}>
          <a href="#x">View result</a>
        </Button>,
      );
      const link = screen.getByRole("link", { name: "View result" });
      expect(link).toHaveAttribute("aria-busy", "true");
      await user.click(link);
      expect(onClick).not.toHaveBeenCalled();
    });
  });
});
