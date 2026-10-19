import { memo, useRef, useState } from 'react';
import { ATTACKS, CATEGORY_LABELS, type Txn } from '../api.ts';
import { useStore } from '../store.ts';
import { clock, money } from '../format.ts';

const VISIBLE = 60;

type Filter = 'all' | 'declined' | 'fraud';

/**
 * The live feed, printed like card-terminal receipt paper. A declined
 * payment gets a red rubber stamp; a payment that was actually fraud but
 * got through is marked "MISSED", because a model's misses matter as much
 * as its catches.
 */
export function ReceiptTape({ onSelect }: { onSelect: (id: number) => void }) {
  const txns = useStore((state) => state.txns);
  const threshold = useStore((state) => state.threshold);
  const selected = useStore((state) => state.selected);
  const [filter, setFilter] = useState<Filter>('all');
  // The feed holds still while the pointer or focus is on it, so a payment
  // can be clicked without it scrolling away.
  const [held, setHeld] = useState(false);
  const frozen = useRef<{ list: Txn[]; newest: number } | null>(null);

  let shown: Txn[] = [];
  for (let index = txns.length - 1; index >= 0 && shown.length < VISIBLE; index -= 1) {
    const txn = txns[index] as Txn;
    const isDeclined = txn.score > threshold;
    if (filter === 'declined' && !isDeclined) continue;
    if (filter === 'fraud' && !txn.fraud) continue;
    shown.push(txn);
  }
  let waiting = 0;
  if (held) {
    if (!frozen.current) frozen.current = { list: shown, newest: shown[0]?.id ?? 0 };
    const newest = frozen.current.newest;
    waiting = shown.filter((txn) => txn.id > newest).length;
    shown = frozen.current.list;
  } else {
    frozen.current = null;
  }

  return (
    <section className="tape" aria-labelledby="tape-title">
      <header className="tape__head">
        <h2 id="tape-title" className="panel-title">
          Live payments
          {held && <span className="tape__held">Paused{waiting ? ` · ${waiting >= VISIBLE ? `${VISIBLE}+` : waiting} new` : ''}</span>}
        </h2>
        <div className="chips" role="group" aria-label="Show">
          {(['all', 'declined', 'fraud'] as const).map((value) => (
            <button key={value} type="button" aria-pressed={filter === value} className="chip" onClick={() => setFilter(value)}>
              {value === 'all' ? 'All' : value === 'declined' ? 'Declined' : 'Actual fraud'}
            </button>
          ))}
        </div>
      </header>
      <ol
        className="tape__paper"
        aria-live="off"
        onPointerEnter={(event) => event.pointerType === 'mouse' && setHeld(true)}
        onPointerLeave={() => setHeld(false)}
        onFocus={(event) => event.target.matches(':focus-visible') && setHeld(true)}
        onBlur={(event) => !event.currentTarget.contains(event.relatedTarget) && setHeld(false)}
      >
        {shown.map((txn) => (
          <Receipt key={txn.id} txn={txn} declined={txn.score > threshold} active={txn.id === selected} onSelect={onSelect} />
        ))}
        {!shown.length && <li className="tape__empty">Nothing yet. {filter === 'all' ? 'Payments appear as they happen.' : 'Try launching an attack.'}</li>}
      </ol>
    </section>
  );
}

const Receipt = memo(function Receipt({ txn, declined, active, onSelect }: { txn: Txn; declined: boolean; active: boolean; onSelect: (id: number) => void }) {
  const missed = txn.fraud && !declined;
  const falseDecline = declined && !txn.fraud;
  const attack = txn.attack ? ATTACKS[txn.attack - 1] : undefined;
  return (
    <li>
      <button
        type="button"
        className={`receipt${declined ? ' is-declined' : ''}${missed ? ' is-missed' : ''}${active ? ' is-active' : ''}`}
        onClick={() => onSelect(txn.id)}
        aria-label={`${money(txn.amount)} at ${txn.merchant}, ${declined ? 'declined' : 'approved'}${missed ? ', missed fraud' : ''}. Risk ${Math.round(txn.score * 100)} of 100.`}
      >
        <span className="receipt__line">
          <span className="receipt__merchant">{txn.merchant}</span>
          <span className="receipt__amount">{money(txn.amount)}</span>
        </span>
        <span className="receipt__line receipt__meta">
          <span>
            {clock(txn.ts)} · {CATEGORY_LABELS[txn.category]} · ••{txn.last4}
          </span>
          <span className={`receipt__risk risk-${riskBand(txn.score)}`}>{Math.round(txn.score * 100)}</span>
        </span>
        {(attack || falseDecline) && (
          <span className="receipt__line receipt__foot">
            <span className="receipt__truth">{attack ? `Actually ${attack.label.toLowerCase()}` : 'Actually an honest customer'}</span>
            {declined && <span className="stamp">Declined</span>}
            {missed && <span className="stamp stamp--missed">Missed</span>}
          </span>
        )}
      </button>
    </li>
  );
});

function riskBand(score: number) {
  return score > 0.5 ? 'high' : score > 0.1 ? 'mid' : 'low';
}
