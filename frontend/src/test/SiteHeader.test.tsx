import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { SiteHeader } from "../components/SiteHeader";

function renderHeader() {
  return render(
    <MemoryRouter>
      <SiteHeader />
    </MemoryRouter>,
  );
}

describe("SiteHeader", () => {
  it("connects the menu toggle to the primary navigation", () => {
    renderHeader();

    const menu = screen.getByRole("button", { name: "Open navigation" });
    const navigation = screen.getByRole("navigation", { name: "Primary navigation" });

    expect(menu).toHaveAttribute("aria-controls", "primary-navigation");
    expect(navigation).toHaveAttribute("id", "primary-navigation");
  });

  it("closes the open menu with Escape and returns focus to its toggle", () => {
    renderHeader();

    const menu = screen.getByRole("button", { name: "Open navigation" });
    fireEvent.click(menu);
    expect(menu).toHaveAttribute("aria-expanded", "true");

    screen.getByRole("link", { name: "Model" }).focus();
    fireEvent.keyDown(document, { key: "Escape" });

    expect(menu).toHaveAttribute("aria-expanded", "false");
    expect(menu).toHaveAccessibleName("Open navigation");
    expect(menu).toHaveFocus();
  });
});
