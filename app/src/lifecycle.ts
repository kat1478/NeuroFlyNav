/** One cleanup per active view (stop animations, observers) when the route changes. */

let cleanup: (() => void) | undefined;

export function onLeave(callback: () => void): void {
  cleanup = callback;
}

export function leave(): void {
  cleanup?.();
  cleanup = undefined;
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
  return () => observer.disconnect();
}
