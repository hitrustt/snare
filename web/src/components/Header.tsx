import { useStore } from '../store.ts';
import { count, microseconds, money, oneIn, percent, simDate } from '../format.ts';

/** Brand, connection state, the simulated clock and the running scoreboard. */
export function Header({ onAbout }: { onAbout: () => void }) {
  const status = useStore((state) => state.status);
  const now = useStore((state) => state.now);
  const stats = useStore((state) => state.stats);

  const honest = stats ? stats.transactions - stats.fraud : 0;
  return (
    <header className="header">
      <div className="brand">
        <Logo />
        <div>
          <h1 className="brand__name">Snare</h1>
          <p className="brand__tag">Real-time fraud detection on a live payment network</p>
        </div>
      </div>

      <dl className="scoreboard" aria-label="Since this server started">
        <Stat label="Payments scored" value={stats ? count(stats.transactions) : '–'} />
        <Stat label="Fraud caught" value={stats && stats.fraud ? percent(stats.fraud_caught / stats.fraud, 0) : '–'} hint={stats ? `${count(stats.fraud_caught)} of ${count(stats.fraud)}` : undefined} />
        <Stat label="Fraud stopped" value={stats ? money(stats.fraud_dollars_caught, true) : '–'} hint={stats ? `of ${money(stats.fraud_dollars, true)}` : undefined} />
        <Stat label="Honest declined" value={stats && honest ? oneIn(stats.false_declines / honest) : '–'} hint="payments" />
        <Stat label="Decision time" value={stats ? microseconds(stats.p50_us) : '–'} hint={stats ? `p99 ${microseconds(stats.p99_us)}` : undefined} />
      </dl>

      <div className="header__side">
        <span className={`status status--${status}`}>
          <span className="status__dot" aria-hidden="true" />
          {status === 'live' ? simDate(now) : status === 'connecting' ? 'Waking the server…' : 'Reconnecting…'}
        </span>
        <span className="visually-hidden" role="status">
          {status === 'live' ? 'Connected' : status === 'connecting' ? 'Connecting' : 'Reconnecting'}
        </span>
        <button type="button" className="button button--ghost" onClick={onAbout}>
          How it works
        </button>
      </div>
    </header>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd>
        <span className="stat__value">{value}</span>
        {hint && <span className="stat__hint">{hint}</span>}
      </dd>
    </div>
  );
}

export function Logo() {
  return (
    <svg className="logo" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="currentColor" />
      <path d="M8 11.5h16M8 16h10M8 20.5h6" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="23" cy="20.5" r="3.4" fill="#ff5a4e" />
    </svg>
  );
}
