export function recordDecisionBoundary(audit, boundary, onBoundary) {
  const immutable = structuredClone(boundary);
  audit.push(immutable);
  onBoundary?.(structuredClone(immutable));
}
