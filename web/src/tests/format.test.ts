import { describe, expect, it } from "vitest";
import { formatDuration, formatScore, statusLabel } from "@/lib/format";

describe("formatDuration", () => {
  it("renders sub-second durations in milliseconds", () => {
    expect(formatDuration(0.42)).toBe("420ms");
  });

  it("renders seconds with one decimal", () => {
    expect(formatDuration(3.256)).toBe("3.3s");
  });

  it("renders minutes with zero-padded seconds", () => {
    expect(formatDuration(83)).toBe("1m 23s");
  });

  it("renders hours with zero-padded minutes", () => {
    expect(formatDuration(3 * 3600 + 5 * 60)).toBe("3h 05m");
  });
});

describe("formatScore", () => {
  it("keeps three decimals", () => {
    expect(formatScore(0.73456)).toBe("0.735");
  });

  it("renders invalid numbers as a dash", () => {
    expect(formatScore(Number.NaN)).toBe("—");
  });
});

describe("statusLabel", () => {
  it("maps known statuses to Chinese labels", () => {
    expect(statusLabel("awaiting_feedback")).toBe("待反馈");
    expect(statusLabel("completed")).toBe("已完成");
  });

  it("passes through unknown statuses", () => {
    expect(statusLabel("mystery")).toBe("mystery");
  });
});
