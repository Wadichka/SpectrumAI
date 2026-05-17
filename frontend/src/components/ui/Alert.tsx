import { cva, type VariantProps } from "class-variance-authority";
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from "lucide-react";
import { useState, type HTMLAttributes, type ReactNode } from "react";

import { cn } from "@/lib/cn";

const alertVariants = cva("flex items-start gap-3 rounded-md border p-4 text-sm", {
  variants: {
    variant: {
      info: "border-primary bg-primary-muted text-primary",
      success: "border-success bg-green-50 text-success",
      warning: "border-warning bg-amber-50 text-warning",
      error: "border-danger bg-red-50 text-danger",
    },
  },
  defaultVariants: { variant: "info" },
});

const VARIANT_ICON = {
  info: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  error: XCircle,
} as const;

interface AlertProps
  extends Omit<HTMLAttributes<HTMLDivElement>, "title">,
    VariantProps<typeof alertVariants> {
  title?: ReactNode;
  closable?: boolean;
}

export default function Alert({
  variant = "info",
  title,
  closable = false,
  className,
  children,
  ...rest
}: AlertProps) {
  const [open, setOpen] = useState(true);
  if (!open) return null;
  const Icon = VARIANT_ICON[variant ?? "info"];
  return (
    <div role="alert" className={cn(alertVariants({ variant }), className)} {...rest}>
      <Icon className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
      <div className="flex-1">
        {title ? <p className="mb-1 font-semibold">{title}</p> : null}
        <div className="text-current">{children}</div>
      </div>
      {closable ? (
        <button
          type="button"
          aria-label="Закрыть"
          onClick={() => setOpen(false)}
          className="rounded p-1 text-current opacity-70 transition-opacity hover:opacity-100"
        >
          <X className="size-4" aria-hidden="true" />
        </button>
      ) : null}
    </div>
  );
}
