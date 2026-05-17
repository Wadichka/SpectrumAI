import { cva, type VariantProps } from "class-variance-authority";
import type { HTMLAttributes } from "react";

import { cn } from "@/lib/cn";

const spinnerVariants = cva("inline-block animate-spin rounded-full border-2 border-current", {
  variants: {
    size: {
      sm: "h-4 w-4 border-t-transparent",
      md: "h-6 w-6 border-t-transparent",
      lg: "h-8 w-8 border-t-transparent",
    },
  },
  defaultVariants: { size: "md" },
});

interface SpinnerProps extends HTMLAttributes<HTMLSpanElement>, VariantProps<typeof spinnerVariants> {
  label?: string;
}

export default function Spinner({ className, size, label, ...rest }: SpinnerProps) {
  return (
    <span
      role="status"
      aria-label={label ?? "Загрузка"}
      className={cn(spinnerVariants({ size }), className)}
      {...rest}
    />
  );
}
