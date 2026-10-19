const START = Date.UTC(2026, 9, 5, 0, 0, 0); // the simulation's day 0, a Monday

const clockFormat = new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'UTC' });
const dayFormat = new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric', timeZone: 'UTC' });

export const clock = (simSeconds: number) => clockFormat.format(START + simSeconds * 1000);
export const simDate = (simSeconds: number) => `${dayFormat.format(START + simSeconds * 1000)}, ${clock(simSeconds)}`;

export function money(value: number, compact = false): string {
  if (compact && value >= 10_000) return `$${(value / 1000).toFixed(value >= 100_000 ? 0 : 1)}k`;
  return value.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: value >= 1000 ? 0 : 2 });
}

export const count = (value: number) => value.toLocaleString('en-US');

export function percent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '–';
  return `${(value * 100).toFixed(digits)}%`;
}

/** "1 in 240" style rates read better than 0.42% for rare events. */
export function oneIn(rate: number): string {
  if (rate <= 0) return 'none';
  const n = Math.round(1 / rate);
  return `1 in ${n.toLocaleString('en-US')}`;
}

export function microseconds(us: number): string {
  return us >= 1000 ? `${(us / 1000).toFixed(1)} ms` : `${Math.round(us)} µs`;
}
