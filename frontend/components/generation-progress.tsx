"use client";

import { useEffect, useRef, useState } from "react";

import { generationSteps, type GenerationStep } from "../src/domain/generationSteps";
import type { PipelineState } from "../src/domain/pipeline";
import { Card } from "./ui/card";

function formatElapsed(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function useElapsedSeconds(): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const startedAt = Date.now();
    const id = window.setInterval(() => {
      setSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, []);
  return seconds;
}

function StepBar({ step }: { step: GenerationStep }) {
  const determinate = step.percent !== null;
  return (
    <div
      role="progressbar"
      aria-label={step.label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={step.percent ?? undefined}
      className="bg-border relative mt-2 h-1.5 overflow-hidden rounded-full"
    >
      <div
        className={
          determinate
            ? "bg-primary h-full rounded-full transition-all duration-300"
            : "progress-indeterminate bg-primary h-full rounded-full"
        }
        style={determinate ? { width: `${step.percent}%` } : undefined}
      />
    </div>
  );
}

function StepRow({ step, index }: { step: GenerationStep; index: number }) {
  const marker = step.status === "done" ? "✓" : String(index + 1);
  return (
    <li aria-current={step.status === "active" ? "step" : undefined} className="flex gap-3">
      <span
        aria-hidden="true"
        className={
          step.status === "pending"
            ? "border-border text-muted-foreground flex h-6 w-6 shrink-0 items-center justify-center rounded-full border font-mono text-xs"
            : "bg-primary text-primary-foreground flex h-6 w-6 shrink-0 items-center justify-center rounded-full font-mono text-xs font-bold"
        }
      >
        {marker}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <span className={step.status === "pending" ? "text-muted-foreground" : "text-foreground"}>
            {step.label}
          </span>
          <span className="text-muted-foreground font-mono text-xs">
            {step.status === "done" ? (
              "Done"
            ) : step.status === "pending" ? (
              "Waiting"
            ) : step.percent !== null ? (
              <span className="text-primary">{step.percent}%</span>
            ) : null}
          </span>
        </div>
        {step.status === "active" ? <StepBar step={step} /> : null}
      </div>
    </li>
  );
}

/**
 * Fills the results slot while a run is in progress. Not a live region: the
 * Generate button's spinner already announces status changes, and a second
 * announcer (or a ticking timer) would just repeat it.
 */
export function GenerationProgress<A>({ state }: { state: PipelineState<A> }) {
  const steps = generationSteps(state);
  if (steps.length === 0) return null;
  // Remount per run so the timer restarts for each generation.
  return <ProgressCard key={"runId" in state ? state.runId : 0} steps={steps} />;
}

function ProgressCard({ steps }: { steps: GenerationStep[] }) {
  const elapsed = useElapsedSeconds();
  const generating = steps.find((s) => s.id === "generate")?.status === "active";
  const ref = useRef<HTMLDivElement>(null);
  // Below the lg breakpoint this card stacks under the controls, off-screen.
  useEffect(() => {
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    ref.current?.scrollIntoView?.({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
  }, []);
  return (
    <Card className="flex min-h-[520px] flex-col justify-start lg:justify-center" variant="muted">
      <div ref={ref} data-testid="generation-progress" className="mx-auto w-full max-w-sm">
        <p className="text-foreground font-display text-lg">Working on your variants</p>
        <ol className="mt-6 space-y-5 text-sm">
          {steps.map((step, index) => (
            <StepRow key={step.id} step={step} index={index} />
          ))}
        </ol>
        <p className="text-muted-foreground mt-8 font-mono text-xs">
          Elapsed {formatElapsed(elapsed)}
        </p>
        {generating ? (
          <p className="text-muted-foreground mt-2 text-xs">
            Generation can take a while on the free server. You can cancel at any time.
          </p>
        ) : null}
      </div>
    </Card>
  );
}
