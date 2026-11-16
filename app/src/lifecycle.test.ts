import { describe, expect, it, vi } from "vitest";
import { Scope } from "./lifecycle";

describe("Scope", () => {
  it("runs cleanups once when the view is left", () => {
    const scope = new Scope();
    const cleanup = vi.fn();
    scope.onLeave(cleanup);
    expect(scope.active).toBe(true);
    scope.dispose();
    scope.dispose();
    expect(cleanup).toHaveBeenCalledTimes(1);
    expect(scope.active).toBe(false);
  });

  it("cleans up at once when a stale view registers after it was left", () => {
    const scope = new Scope();
    scope.dispose();
    const cleanup = vi.fn();
    scope.onLeave(cleanup);
    expect(cleanup).toHaveBeenCalledTimes(1);
  });
});
