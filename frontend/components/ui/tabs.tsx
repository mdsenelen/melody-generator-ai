"use client";

/**
 * Tabs: switch between views of the same subject. Radix Tabs + cva; conventions in ./variants.
 *
 * Usage
 *   <Tabs defaultValue="notes" size="sm">
 *     <TabsList aria-label="Analysis views">
 *       <TabsTrigger value="notes">Notes</TabsTrigger>
 *       <TabsTrigger value="chords" loading>Chords</TabsTrigger>
 *       <TabsTrigger value="key" disabled>Key</TabsTrigger>
 *     </TabsList>
 *     <TabsContent value="notes">...</TabsContent>
 *     <TabsContent value="chords" loading>...</TabsContent>
 *   </Tabs>
 *   // controlled: <Tabs value={tab} onValueChange={setTab}>
 *
 * Accessible name: `TabsList` requires `aria-label` or `aria-labelledby` (a compile error without).
 * Keyboard (from Radix): Arrow keys move and select with wrap, Home/End jump, disabled tabs are
 * skipped; `aria-selected` / `aria-controls` / `aria-labelledby` are wired by Radix.
 *
 * States
 *   selected        an accent bar under the tab AND stronger text, never colour alone
 *   hover / active  `hover:` / `active:` text and background steps
 *   focus-visible   FOCUS_RING_INSET on triggers (the list clips its bar); FOCUS_RING on the panel
 *   disabled        native `disabled`; skipped by arrow keys, not activatable
 *   loading         `aria-busy="true"` on the trigger (a pulsing dot) and, via `TabsContent loading`,
 *                   on its panel. The trigger stays enabled so the user can still look at the panel;
 *                   the caller owns the live-region status text.
 *   error           not applicable: a tab is navigation, not a value. Show the failure inside the
 *                   panel with an ErrorState.
 */
import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cva } from "class-variance-authority";

import { cn } from "./cn";
import { DISABLED, FOCUS_RING, FOCUS_RING_INSET, LOADING, type Exhaustive } from "./variants";

export type TabsSize = "sm" | "md";

const TabsSizeContext = React.createContext<TabsSize>("md");

const TRIGGER_SIZES: Exhaustive<TabsSize> = {
  sm: "h-8 px-3 text-2xs",
  md: "h-10 px-4 text-xs",
};

const tabsTrigger = cva(
  [
    "relative inline-flex shrink-0 items-center justify-center gap-2 font-display tracking-wide whitespace-nowrap uppercase",
    "text-text-muted transition-colors duration-(--duration-fast) ease-standard",
    "hover:bg-surface-raised hover:text-text active:bg-surface-muted",
    "data-[state=active]:text-text data-[state=active]:font-semibold",
    "after:absolute after:inset-x-0 after:bottom-0 after:h-0.5 after:bg-transparent data-[state=active]:after:bg-accent",
    FOCUS_RING_INSET,
    DISABLED,
    LOADING,
  ],
  { variants: { size: TRIGGER_SIZES }, defaultVariants: { size: "md" } },
);

export const Tabs = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Root> & { size?: TabsSize }
>(function Tabs({ size = "md", className, ...rest }, ref) {
  return (
    <TabsSizeContext.Provider value={size}>
      <TabsPrimitive.Root ref={ref} className={cn("flex flex-col", className)} {...rest} />
    </TabsSizeContext.Provider>
  );
});

type NameProps = { "aria-label": string } | { "aria-labelledby": string };

export type TabsListProps = Omit<
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>,
  "aria-label" | "aria-labelledby"
> &
  NameProps;

export const TabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  TabsListProps
>(function TabsList({ className, ...rest }, ref) {
  return (
    <TabsPrimitive.List
      ref={ref}
      className={cn("border-border flex items-stretch overflow-x-auto border-b", className)}
      {...rest}
    />
  );
});

export type TabsTriggerProps = React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger> & {
  loading?: boolean;
};

export const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  TabsTriggerProps
>(function TabsTrigger({ className, children, loading = false, ...rest }, ref) {
  const size = React.useContext(TabsSizeContext);
  return (
    <TabsPrimitive.Trigger
      ref={ref}
      aria-busy={loading ? true : undefined}
      className={cn(tabsTrigger({ size }), className)}
      {...rest}
    >
      {children}
      {loading ? (
        <span
          aria-hidden="true"
          className="bg-accent size-1.5 rounded-full motion-safe:animate-pulse"
        />
      ) : null}
    </TabsPrimitive.Trigger>
  );
});

export type TabsContentProps = React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content> & {
  loading?: boolean;
};

export const TabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  TabsContentProps
>(function TabsContent({ className, loading = false, ...rest }, ref) {
  return (
    <TabsPrimitive.Content
      ref={ref}
      aria-busy={loading ? true : undefined}
      className={cn("pt-3", FOCUS_RING, LOADING, className)}
      {...rest}
    />
  );
});
