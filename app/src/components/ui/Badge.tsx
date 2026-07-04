/**
 * Badge / chip components for status indicators, tags, counts.
 */

type BadgeVariant = "amber" | "success" | "danger" | "neutral" | "info";

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
}

const variantClasses: Record<BadgeVariant, string> = {
  amber:   "bg-amber-500/15  border-amber-500/35  text-amber-400",
  success: "bg-emerald-500/15 border-emerald-500/30 text-emerald-400",
  danger:  "bg-red-500/15    border-red-500/30    text-red-400",
  info:    "bg-blue-500/15   border-blue-500/30   text-blue-400",
  neutral: "bg-zinc-700/50   border-zinc-600/50   text-zinc-400",
};

export function Badge({
  children,
  variant = "neutral",
  className = "",
}: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full
                  text-[10px] font-semibold tracking-wide border
                  ${variantClasses[variant]} ${className}`}
    >
      {children}
    </span>
  );
}

export function PulseBadge({
  label,
  variant = "success",
}: {
  label: string;
  variant?: BadgeVariant;
}) {
  return (
    <Badge variant={variant} className="gap-1.5">
      <span
        className={`w-1.5 h-1.5 rounded-full animate-ping ${
          variant === "success" ? "bg-emerald-400" : "bg-current"
        }`}
        aria-hidden
      />
      {label}
    </Badge>
  );
}
