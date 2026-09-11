import { useId, type InputHTMLAttributes } from "react";

import { Label } from "./text";

type SliderProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (value: number) => void;
  disabled?: boolean;
} & Pick<InputHTMLAttributes<HTMLInputElement>, "aria-label">;

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  unit = "",
  onChange,
  disabled = false,
  ...rest
}: SliderProps) {
  const id = useId();
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className={disabled ? "space-y-2 opacity-40" : "space-y-2"}>
      <div className="flex justify-between">
        <Label>
          <label htmlFor={id}>{label}</label>
        </Label>
        <span className="text-primary font-mono text-[11px]">
          {value}
          {unit}
        </span>
      </div>
      <div className="bg-border relative h-px">
        <div className="bg-primary absolute top-0 left-0 h-full" style={{ width: `${pct}%` }} />
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(Number(event.target.value))}
          className="absolute inset-0 w-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
          style={{ height: 20, top: -10 }}
          {...rest}
        />
      </div>
    </div>
  );
}

/** A compact 0.3-2.0 temperature dial, one per generated variant. */
export function TemperatureInput({
  label,
  value,
  onChange,
  disabled = false,
}: {
  label?: string;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const pct = ((value - 0.3) / 1.7) * 100;
  const warmth = value > 1.2 ? "text-amber-400" : value < 0.7 ? "text-sky-400" : "text-primary";

  return (
    <div className={disabled ? "flex items-center gap-2 opacity-40" : "flex items-center gap-2"}>
      {label ? (
        <label htmlFor={id} className="text-muted-foreground w-5 shrink-0 font-mono text-[10px]">
          {label}
        </label>
      ) : null}
      <span className={`w-7 shrink-0 text-right font-mono text-[11px] ${warmth}`}>
        {value.toFixed(1)}
      </span>
      <div className="bg-border relative h-px flex-1">
        <div className="bg-primary absolute top-0 left-0 h-full" style={{ width: `${pct}%` }} />
        <input
          id={id}
          type="range"
          min={0.3}
          max={2.0}
          step={0.1}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(Number(event.target.value))}
          className="absolute inset-0 w-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
          style={{ height: 20, top: -10 }}
        />
      </div>
    </div>
  );
}
