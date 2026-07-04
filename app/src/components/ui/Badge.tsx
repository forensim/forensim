import { cn } from "../../lib/utils";

type BadgeVariant = "success" | "error" | "warning" | "info" | "neutral";

export interface BadgeProps {
  label: string;
  variant?: BadgeVariant;
  className?: string;
}

const variantStyles: Record<BadgeVariant, string> = {
  success: "bg-green-900/50 text-green-400 border-green-800/50",
  error: "bg-red-900/50 text-red-400 border-red-800/50",
  warning: "bg-amber-900/50 text-amber-400 border-amber-800/50",
  info: "bg-blue-900/50 text-blue-400 border-blue-800/50",
  neutral: "bg-gray-800/50 text-gray-400 border-gray-700/50",
};

export function Badge({ label, variant = "neutral", className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium",
        variantStyles[variant],
        className
      )}
    >
      {label}
    </span>
  );
}

export default Badge;
