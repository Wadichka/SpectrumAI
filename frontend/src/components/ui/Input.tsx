import { forwardRef, type InputHTMLAttributes, type ReactNode } from "react";

import { cn } from "@/lib/cn";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  iconLeft?: ReactNode;
}

const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { label, error, iconLeft, className, id, ...rest },
  ref,
) {
  const inputId = id ?? rest.name;
  return (
    <div className="flex flex-col gap-1">
      {label ? (
        <label htmlFor={inputId} className="text-sm font-medium text-ink">
          {label}
        </label>
      ) : null}
      <div className="relative">
        {iconLeft ? (
          <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted">
            {iconLeft}
          </span>
        ) : null}
        <input
          id={inputId}
          ref={ref}
          aria-invalid={error ? "true" : undefined}
          className={cn(
            "w-full rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink",
            "placeholder:text-muted focus:border-primary focus:outline-none",
            "focus:ring-2 focus:ring-primary focus:ring-offset-1",
            "disabled:cursor-not-allowed disabled:opacity-50",
            iconLeft ? "pl-10" : "",
            error ? "border-danger focus:border-danger focus:ring-danger" : "",
            className,
          )}
          {...rest}
        />
      </div>
      {error ? (
        <span role="alert" className="text-sm text-danger">
          {error}
        </span>
      ) : null}
    </div>
  );
});

export default Input;
