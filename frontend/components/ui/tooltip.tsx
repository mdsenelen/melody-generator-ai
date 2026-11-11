"use client";

/**
 * Tooltip: a short text hint for a control. Radix Tooltip; conventions in ./variants.
 *
 * Usage
 *   <Tooltip content="Play from the start"><Button>Play</Button></Tooltip>
 *   <Tooltip content="Space" side="bottom"><IconButton label="Play" icon={<PlayIcon />} /></Tooltip>
 *
 * A tooltip SUPPLEMENTS the accessible name, it never replaces it: an IconButton still needs its
 * `label`. Radix links the trigger to the `role="tooltip"` node with `aria-describedby`.
 * `content` is `string` on purpose: a tooltip cannot be focused or hovered reliably, so it must
 * never hold a link or button. Put interactive content in a Dialog instead.
 *
 * Provider: self-providing. `<Tooltip>` mounts its own Radix provider unless it is inside a
 * `<TooltipProvider>`, so a lone Tooltip works with no setup. Wrap a toolbar (or the app) in
 * `<TooltipProvider>` only to share the open-delay across neighbours (moving between triggers
 * then opens instantly). Delay: `TOOLTIP_DELAY_MS`, one constant; `delayDuration` overrides it.
 *
 * States
 *   hover / focus   opens after the delay on pointer hover and on keyboard focus
 *   Escape          closes it
 *   motion          fade via the duration tokens (collapses under prefers-reduced-motion)
 *   disabled, active, loading, error   not applicable: a tooltip is not itself a control. A
 *                   disabled trigger gets no tooltip (a disabled button fires no pointer or focus
 *                   events); say why it is disabled in visible text instead.
 */
import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { cva } from "class-variance-authority";

import { cn } from "./cn";
import type { Exhaustive } from "./variants";

/** Open delay for every tooltip in the app. */
export const TOOLTIP_DELAY_MS = 400;

export type TooltipTone = "default" | "mono";

const TONES: Exhaustive<TooltipTone> = {
  default: "font-sans",
  /** Times, frequencies and key bindings read as instrument values. */
  mono: "font-mono tabular-nums",
};

const tooltipContent = cva(
  [
    "bg-surface-overlay text-text border-border-strong text-2xs z-50 max-w-60 rounded-[var(--radius)] border px-2 py-1 shadow-popover",
    "data-[state=delayed-open]:animate-[ui-fade-in_var(--duration-fast)_var(--ease-standard)]",
    "data-[state=instant-open]:animate-[ui-fade-in_var(--duration-instant)_var(--ease-standard)]",
    "data-[state=closed]:animate-[ui-fade-out_var(--duration-fast)_var(--ease-standard)]",
  ],
  { variants: { tone: TONES }, defaultVariants: { tone: "default" } },
);

const HasProviderContext = React.createContext(false);

export function TooltipProvider({
  delayDuration = TOOLTIP_DELAY_MS,
  children,
  ...rest
}: React.ComponentProps<typeof TooltipPrimitive.Provider>) {
  return (
    <HasProviderContext.Provider value>
      <TooltipPrimitive.Provider delayDuration={delayDuration} {...rest}>
        {children}
      </TooltipPrimitive.Provider>
    </HasProviderContext.Provider>
  );
}

export type TooltipProps = {
  /** Plain text only. */
  content: string;
  children: React.ReactElement;
  tone?: TooltipTone;
  side?: TooltipPrimitive.TooltipContentProps["side"];
  align?: TooltipPrimitive.TooltipContentProps["align"];
  delayDuration?: number;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  className?: string;
};

export function Tooltip({
  content,
  children,
  tone,
  side = "top",
  align,
  delayDuration = TOOLTIP_DELAY_MS,
  className,
  ...root
}: TooltipProps) {
  const hasProvider = React.useContext(HasProviderContext);
  const body = (
    <TooltipPrimitive.Root delayDuration={delayDuration} {...root}>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          side={side}
          align={align}
          sideOffset={6}
          collisionPadding={8}
          className={cn(tooltipContent({ tone }), className)}
        >
          {content}
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
  return hasProvider ? body : <TooltipProvider>{body}</TooltipProvider>;
}
