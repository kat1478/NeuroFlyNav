/** Each rendered view gets a scope; leaving the view disposes it and runs its cleanups. */

export class Scope {
  private cleanups: (() => void)[] = [];
  private disposed = false;

  /** Register a cleanup; runs at once if the view was already left (stale render). */
  onLeave(callback: () => void): void {
    if (this.disposed) callback();
    else this.cleanups.push(callback);
  }

  get active(): boolean {
    return !this.disposed;
  }

  dispose(): void {
    this.disposed = true;
    for (const cleanup of this.cleanups.splice(0)) cleanup();
  }
}

/** Re-run `draw` when the element's width changes (not on height changes it causes). */
export function onWidthChange(element: HTMLElement, draw: () => void): () => void {
  let width = element.clientWidth;
  let frame = 0;
  const observer = new ResizeObserver(() => {
    if (element.clientWidth === width) return;
    width = element.clientWidth;
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(draw);
  });
  observer.observe(element);
  return () => {
    cancelAnimationFrame(frame);
    observer.disconnect();
  };
}
