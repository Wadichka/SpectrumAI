import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import Button from "@/components/ui/Button";

describe("Button", () => {
  it("рендерит дочерний текст", () => {
    render(<Button>Confirm</Button>);
    expect(screen.getByRole("button", { name: "Confirm" })).toBeInTheDocument();
  });

  it("показывает Spinner и выставляет aria-busy при loading", () => {
    render(<Button loading>Loading</Button>);
    const button = screen.getByRole("button", { name: /loading/i });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("status", { hidden: true })).toBeInTheDocument();
  });

  it("применяет вариант destructive", () => {
    render(<Button variant="destructive">Delete</Button>);
    const button = screen.getByRole("button", { name: "Delete" });
    expect(button.className).toContain("bg-danger");
  });

  it("блокирует клик при disabled", () => {
    render(
      <Button disabled aria-label="disabled-action">
        Inactive
      </Button>,
    );
    const button = screen.getByRole("button", { name: "disabled-action" });
    expect(button).toBeDisabled();
  });
});
