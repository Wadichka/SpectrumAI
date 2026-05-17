const UNITS = ["B", "KB", "MB", "GB"] as const;

export function formatBytes(size: number): string {
  if (!Number.isFinite(size) || size < 0) {
    return "—";
  }
  let value = size;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < UNITS.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const formatted = unitIndex === 0 ? value.toFixed(0) : value.toFixed(1);
  return `${formatted} ${UNITS[unitIndex]}`;
}
