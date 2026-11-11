"use client";

/**
 * Dialog: a modal for one focused task or confirmation. Radix Dialog + cva.
 *
 * Usage
 *   <Dialog>
 *     <DialogTrigger asChild><Button>Delete take</Button></DialogTrigger>
 *     <DialogContent title="Delete this take?" description="This cannot be undone." tone="destructive">
 *       <DialogFooter>
 *         <DialogClose asChild><Button variant="ghost">Keep take</Button></DialogClose>
 *         <Button variant="destructive" onClick={remove}>Delete take</Button>
 *       </DialogFooter>
 *     </DialogContent>
 *   </Dialog>
 *   // controlled: <Dialog open={open} onOpenChange={setOpen}>
 *
 * Title decision: `DialogContent` takes a REQUIRED `title` prop and renders the Radix title
 * itself, so an untitled dialog does not compile and the dialog is always named
 * (`aria-labelledby`). `description` is optional; when omitted `aria-describedby` is explicitly
 * cleared (no Radix warning). `DialogTitle` / `DialogDescription` are exported for the rare custom
 * header, but the wrapper is the supported path.
 *
 * Modal behaviour (Radix): focus trapped and moved in on open, Escape and overlay click close,
 * focus returns to the trigger, the rest of the page is `aria-hidden` and inert.
 *
 * States
 *   focus-visible   FOCUS_RING on the close button (IconButton) and on anything inside
 *   disabled        not applicable to the dialog itself; its buttons carry their own
 *   loading         `loading` sets `aria-busy` and blocks Escape, overlay click and the close button
 *                   so a pending action cannot be dismissed half-done; the caller owns the live region
 *   error           the `destructive` tone (danger border and title mark) for confirmations; for a
 *                   failed action put an ErrorState inside the body
 *   enter / exit    fade + 2% scale via the duration tokens (collapses under reduced motion)
 *
 * Layout: width is `min(100vw - 1rem, 28rem)`, height capped at the viewport with an internally
 * scrolling body, so it holds at 320px wide.
 */
import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { cva } from "class-variance-authority";

import { IconButton } from "./icon-button";
import { cn } from "./cn";
import type { Exhaustive } from "./variants";

export type DialogTone = "default" | "destructive";

const TONES: Exhaustive<DialogTone> = {
  default: "border-border-strong",
  destructive: "border-danger-border",
};

const dialogContent = cva(
  [
    "bg-surface-overlay text-text shadow-overlay fixed top-1/2 left-1/2 z-50 flex max-h-[calc(100dvh-1rem)] w-[calc(100vw-1rem)] max-w-md -translate-x-1/2 -translate-y-1/2 flex-col rounded-[var(--radius)] border",
    "data-[state=open]:animate-[ui-pop-in_var(--duration-base)_var(--ease-standard)]",
    "data-[state=closed]:animate-[ui-pop-out_var(--duration-fast)_var(--ease-standard)]",
  ],
  { variants: { tone: TONES }, defaultVariants: { tone: "default" } },
);

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;
export const DialogClose = DialogPrimitive.Close;

export const DialogTitle = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(function DialogTitle({ className, ...rest }, ref) {
  return (
    <DialogPrimitive.Title
      ref={ref}
      className={cn("font-display text-sm font-semibold tracking-wide", className)}
      {...rest}
    />
  );
});

export const DialogDescription = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>
>(function DialogDescription({ className, ...rest }, ref) {
  return (
    <DialogPrimitive.Description
      ref={ref}
      className={cn("text-text-muted mt-1 text-xs", className)}
      {...rest}
    />
  );
});

export type DialogContentProps = Omit<
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>,
  "title" | "aria-label" | "aria-labelledby"
> & {
  /** Accessible name. Required. */
  title: string;
  description?: string;
  tone?: DialogTone;
  loading?: boolean;
};

export const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  DialogContentProps
>(function DialogContent(
  {
    title,
    description,
    tone = "default",
    loading = false,
    className,
    children,
    onEscapeKeyDown,
    onInteractOutside,
    ...rest
  },
  ref,
) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/60 data-[state=closed]:animate-[ui-fade-out_var(--duration-fast)_var(--ease-standard)] data-[state=open]:animate-[ui-fade-in_var(--duration-base)_var(--ease-standard)]" />
      <DialogPrimitive.Content
        ref={ref}
        data-tone={tone}
        aria-busy={loading ? true : undefined}
        onEscapeKeyDown={(event) => {
          if (loading) event.preventDefault();
          onEscapeKeyDown?.(event);
        }}
        onInteractOutside={(event) => {
          if (loading) event.preventDefault();
          onInteractOutside?.(event);
        }}
        className={cn(dialogContent({ tone }), className)}
        {...(description ? {} : { "aria-describedby": undefined })}
        {...rest}
      >
        <div className="border-border flex items-start justify-between gap-3 border-b px-4 py-3">
          <div className="min-w-0">
            <DialogTitle className={cn(tone === "destructive" && "text-danger")}>
              {title}
            </DialogTitle>
            {description ? <DialogDescription>{description}</DialogDescription> : null}
          </div>
          <DialogPrimitive.Close asChild>
            <IconButton
              label="Close"
              size="sm"
              variant="ghost"
              disabled={loading}
              icon={
                <svg
                  viewBox="0 0 12 12"
                  width="12"
                  height="12"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <path d="M2 2l8 8M10 2l-8 8" />
                </svg>
              }
            />
          </DialogPrimitive.Close>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 text-sm">{children}</div>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
});

export function DialogFooter({ className, ...rest }: React.ComponentPropsWithoutRef<"div">) {
  return (
    <div
      className={cn("mt-4 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end", className)}
      {...rest}
    />
  );
}
