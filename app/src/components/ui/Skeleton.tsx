/**
 * Skeleton loading placeholder components.
 *
 * Usage:
 *   <Skeleton className="h-4 w-48" />
 *   <SkeletonCard lines={3} />
 */

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = "" }: SkeletonProps) {
  return (
    <div
      className={`skeleton rounded ${className}`}
      aria-hidden="true"
    />
  );
}

export function SkeletonCard({ lines = 2 }: { lines?: number }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-3 animate-fade-up">
      <Skeleton className="h-3 w-24" />
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={`h-2.5 ${i === lines - 1 ? "w-3/4" : "w-full"}`}
        />
      ))}
    </div>
  );
}

export function SkeletonRow() {
  return (
    <div className="flex items-center gap-3 py-2">
      <Skeleton className="h-3 w-3 rounded-full" />
      <Skeleton className="h-2.5 flex-1" />
      <Skeleton className="h-2.5 w-16" />
    </div>
  );
}
