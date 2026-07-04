import { type ReactNode } from "react";
import { cn } from "../../lib/utils";

export interface CardProps {
  children: ReactNode;
  title?: string;
  className?: string;
}

export function Card({ children, title, className }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-gray-800 bg-gray-900 p-4",
        className
      )}
    >
      {title && (
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}

export default Card;
