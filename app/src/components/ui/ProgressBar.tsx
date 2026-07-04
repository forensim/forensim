/**
 * Animated progress bar with optional striped pattern for indeterminate state.
 */

interface ProgressBarProps {
  /** Value 0–100. If undefined, shows indeterminate animation. */
  value?: number;
  /** Colour variant. */
  variant?: "amber" | "success" | "info" | "danger";
  /** Show percentage label inside the bar. */
  showLabel?: boolean;
  /** Additional wrapper class. */
  className?: string;
  /** Height class (default: h-2). */
  height?: string;
}

const trackColors: Record<string, string> = {
  amber:   "bg-amber-500",
  success: "bg-emerald-500",
  info:    "bg-blue-500",
  danger:  "bg-red-500",
};

export function ProgressBar({
  value,
  variant = "amber",
  showLabel = false,
  className = "",
  height = "h-2",
}: ProgressBarProps) {
  const isIndeterminate = value === undefined;
  const pct = isIndeterminate ? 100 : Math.min(100, Math.max(0, value));
  const fill = trackColors[variant] ?? trackColors.amber;

  return (
    <div
      className={`w-full bg-zinc-800 rounded-full overflow-hidden ${height} ${className}`}
      role="progressbar"
      aria-valuenow={isIndeterminate ? undefined : pct}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className={`${height} ${fill} rounded-full transition-all duration-300
                    ${isIndeterminate ? "progress-bar-stripe animate-pulse" : ""}`}
        style={{ width: `${pct}%` }}
      >
        {showLabel && !isIndeterminate && (
          <span className="sr-only">{pct}%</span>
        )}
      </div>
    </div>
  );
}

/** Compact labelled progress row for step-by-step status lists. */
export function ProgressRow({
  label,
  value,
  variant = "amber",
}: {
  label: string;
  value: number;
  variant?: "amber" | "success" | "info" | "danger";
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-zinc-400">{label}</span>
        <span className="font-mono text-zinc-500">{value}%</span>
      </div>
      <ProgressBar value={value} variant={variant} height="h-1.5" />
    </div>
  );
}
