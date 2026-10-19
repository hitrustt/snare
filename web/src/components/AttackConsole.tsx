import { useState } from 'react';
import { ATTACKS, launchAttack, type Attack, type AttackKind } from '../api.ts';
import { store, useStore } from '../store.ts';
import { money } from '../format.ts';

/**
 * Launch a real attack into the shared network. The server builds it from
 * the same playbooks the model was trained against, picks a victim, and
 * feeds the fraudulent payments in among everyone else's.
 */
export function AttackConsole({ onFocusAttack }: { onFocusAttack: (scenario: number) => void }) {
  const attacks = useStore((state) => state.attacks);
  const now = useStore((state) => state.now);
  const status = useStore((state) => state.status);
  const [busy, setBusy] = useState<AttackKind | null>(null);

  const launch = async (kind: AttackKind) => {
    setBusy(kind);
    try {
      const attack = await launchAttack(kind);
      store.notify(`${attack.label} launched against ${attack.accounts === 1 ? 'one account' : `${attack.accounts} accounts`}. Watch the map.`);
    } catch (error) {
      store.notify(error instanceof Error ? error.message : 'The attack could not be launched.');
    } finally {
      setBusy(null);
    }
  };

  const recent = [...attacks].reverse().slice(0, 3);
  return (
    <section className="panel attack" aria-labelledby="attack-title">
      <h2 id="attack-title" className="panel-title">
        Attack the network
      </h2>
      <div className="attack__grid">
        {ATTACKS.map((attack) => (
          <button key={attack.kind} type="button" className={`attack__button attack--${attack.kind}`} disabled={status !== 'live' || busy !== null} onClick={() => launch(attack.kind)}>
            <span className="attack__label">
              <AttackIcon kind={attack.kind} />
              {attack.label}
            </span>
            <span className="attack__summary">{attack.summary}</span>
          </button>
        ))}
      </div>

      <h3 className="subhead">Recent attacks</h3>
      {recent.length === 0 ? (
        <p className="muted">None yet. Launch one above; it plays out in a few seconds.</p>
      ) : (
        <ul className="runs">
          {recent.map((attack) => (
            <AttackRun key={attack.scenario} attack={attack} now={now} onFocus={() => onFocusAttack(attack.scenario)} />
          ))}
        </ul>
      )}
    </section>
  );
}

function AttackRun({ attack, now, onFocus }: { attack: Attack; now: number; onFocus: () => void }) {
  const running = now < attack.end + 60;
  const progress = attack.transactions ? Math.min(1, attack.seen / attack.transactions) : 0;
  const caughtShare = attack.seen ? attack.caught / attack.seen : 0;
  const firstCaught = attack.caught > 0;
  return (
    <li>
      <button type="button" className="run" onClick={onFocus} disabled={attack.seen === 0}>
        <span className="run__top">
          <span className="run__name">
            {attack.label}{' '}
            <span className="run__source">{attack.source === 'visitor' ? 'by a visitor' : 'background'}</span>
          </span>
          <span className={`run__state ${running ? 'is-running' : firstCaught ? 'is-caught' : 'is-missed'}`}>
            {running && attack.seen < attack.transactions ? 'In progress' : firstCaught ? 'Caught' : attack.seen ? 'Got through' : 'Starting'}
          </span>
        </span>
        <span className="run__bar" aria-hidden="true">
          <span className="run__fill" style={{ width: `${progress * 100}%` }}>
            <span className="run__caught" style={{ width: `${caughtShare * 100}%` }} />
          </span>
        </span>
        <span className="run__numbers">
          <span>
            <b>{attack.caught}</b>/{attack.seen} blocked
          </span>
          <span>
            {money(attack.dollars_caught)} of {money(attack.dollars)} stopped
          </span>
        </span>
      </button>
    </li>
  );
}

function AttackIcon({ kind }: { kind: AttackKind }) {
  const paths: Record<AttackKind, string> = {
    card_testing: 'M3 7h18v10H3zM3 10.5h18M6.5 14h3',
    account_takeover: 'M8 11V8a4 4 0 0 1 8 0M5.5 11h13v9h-13zM12 14.5v2',
    skimmed_card: 'M4 6h11v8H4zM9 10h11v8H9z',
    fraud_ring: 'M12 5.5a2 2 0 1 0 0 .01M5.5 17a2 2 0 1 0 0 .01M18.5 17a2 2 0 1 0 0 .01M12 7.5 6.5 15.5M12 7.5l5.5 8M7.5 17h9',
  };
  return (
    <svg viewBox="0 0 24 24" className="icon" aria-hidden="true">
      <path d={paths[kind]} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
