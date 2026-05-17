import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import Card from "@/components/ui/Card";

describe("Card", () => {
  it("рендерит заголовок и контент", () => {
    render(<Card title="Метаданные">тело карточки</Card>);
    expect(screen.getByRole("heading", { name: "Метаданные" })).toBeInTheDocument();
    expect(screen.getByText("тело карточки")).toBeInTheDocument();
  });

  it("отрисовывает actions-блок", () => {
    render(
      <Card title="Заголовок" actions={<button type="button">Действие</button>}>
        body
      </Card>,
    );
    expect(screen.getByRole("button", { name: "Действие" })).toBeInTheDocument();
  });
});
