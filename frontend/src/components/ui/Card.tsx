import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "@/lib/cn";

interface CardProps extends Omit<HTMLAttributes<HTMLDivElement>, "title"> {
  title?: ReactNode;
  actions?: ReactNode;
  padding?: "sm" | "md" | "lg";
}

const PADDING: Record<NonNullable<CardProps["padding"]>, string> = {
  sm: "p-4",
  md: "p-6",
  lg: "p-8",
};

export default function Card({
  title,
  actions,
  padding = "md",
  className,
  children,
  ...rest
}: CardProps) {
  return (
    <section
      className={cn(
        "rounded-lg border border-line bg-surface shadow-sm",
        PADDING[padding],
        className,
      )}
      {...rest}
    >
      {title || actions ? (
        <header className="mb-4 flex items-start justify-between gap-4">
          {title ? <h2 className="text-lg font-semibold text-ink">{title}</h2> : <span />}
          {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
        </header>
      ) : null}
      <div className="text-sm text-ink">{children}</div>
    </section>
  );
}
