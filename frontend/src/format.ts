export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 }).format(value);
}

export function formatMetric(
  value: number | null | undefined,
  digits: number,
  { scale = 1, suffix = "" }: { scale?: number; suffix?: string } = {},
): string {
  return value == null ? "—" : `${(value * scale).toFixed(digits)}${suffix}`;
}
