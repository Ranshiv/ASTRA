/** Shared primitives (ui.tsx). Select is the one covered here directly --
 * it's the sole native <select> replacement in the app (see the AcquirePanel
 * "Provider" dropdown fix), and no existing view test exercised opening or
 * picking an option before this rewrite (confirmed while planning it), so a
 * regression there would otherwise go unnoticed until a human clicked it.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Select } from "@/components/ui";

const OPTIONS = [
  { value: "generic", label: "generic · packet" },
  { value: "gcn", label: "gcn · multimessenger" },
  { value: "alerce", label: "alerce · alert_broker" },
];

describe("Select", () => {
  it("shows the current value on the closed trigger", () => {
    render(<Select label="Provider" value="gcn" options={OPTIONS} onChange={() => {}} />);
    expect(screen.getByText("gcn · multimessenger")).toBeInTheDocument();
  });

  it("opens and lists every option on click", async () => {
    render(<Select label="Provider" value="generic" options={OPTIONS} onChange={() => {}} />);
    fireEvent.keyDown(screen.getByRole("combobox"), { key: "Enter" });

    expect(await screen.findByRole("option", { name: "alerce · alert_broker" }))
      .toBeInTheDocument();
    for (const option of OPTIONS) {
      expect(screen.getByRole("option", { name: option.label })).toBeInTheDocument();
    }
  });

  it("calls onChange with the picked option's value", async () => {
    const onChange = vi.fn();
    render(<Select label="Provider" value="generic" options={OPTIONS} onChange={onChange} />);
    fireEvent.keyDown(screen.getByRole("combobox"), { key: "Enter" });

    fireEvent.click(await screen.findByRole("option", { name: "gcn · multimessenger" }));

    expect(onChange).toHaveBeenCalledWith("gcn");
  });

  it("associates the visible label with the trigger for accessibility", () => {
    render(<Select label="Provider" value="generic" options={OPTIONS} onChange={() => {}} />);
    expect(screen.getByRole("combobox", { name: "Provider" })).toBeInTheDocument();
  });

  describe("an option valued the empty string (e.g. CrossSurveyPanel's 'Automatic')", () => {
    const WITH_DEFAULT = [
      { value: "", label: "Automatic · largest catalogue" },
      { value: "Gaia", label: "Explicit · Gaia" },
    ];

    it("shows its label on the closed trigger instead of rendering blank", () => {
      render(<Select label="Grouping anchor" value="" options={WITH_DEFAULT} onChange={() => {}} />);
      expect(screen.getByText("Automatic · largest catalogue")).toBeInTheDocument();
    });

    it("reports '' to onChange when picked back after a non-empty value", async () => {
      const onChange = vi.fn();
      render(<Select label="Grouping anchor" value="Gaia" options={WITH_DEFAULT} onChange={onChange} />);
      fireEvent.keyDown(screen.getByRole("combobox"), { key: "Enter" });

      fireEvent.click(await screen.findByRole("option", { name: "Automatic · largest catalogue" }));

      expect(onChange).toHaveBeenCalledWith("");
    });
  });
});
