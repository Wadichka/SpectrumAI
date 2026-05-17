import { act, render, screen } from "@testing-library/react";
import { I18nextProvider, useTranslation } from "react-i18next";
import { describe, expect, it } from "vitest";

import i18n from "@/i18n";

function Sample() {
  const { t } = useTranslation();
  return <span>{t("nav.identify")}</span>;
}

describe("i18n", () => {
  it("переключается между русским и английским", async () => {
    render(
      <I18nextProvider i18n={i18n}>
        <Sample />
      </I18nextProvider>,
    );
    expect(screen.getByText("Идентификация")).toBeInTheDocument();
    await act(async () => {
      await i18n.changeLanguage("en");
    });
    expect(screen.getByText("Identify")).toBeInTheDocument();
    await act(async () => {
      await i18n.changeLanguage("ru");
    });
  });
});
