import { useEffect, useRef, useState } from 'react';
import { ATTACKS, CATEGORY_LABELS, fetchCase, unpack, type Case, type Reason, type Txn } from '../api.ts';
import { store, useStore } from '../store.ts';
import { clock, microseconds, money } from '../format.ts';
import { RingGraph } from './RingGraph.tsx';

/**
 * Why a payment was approved or declined: the score, the handful of signals
 * that moved it (from the model's own SHAP values, not a separate guess),
 * and, for rings, the accounts it is linked to.
 */
export function CaseDrawer() {
  const selected = useStore((state) => state.selected);
  const threshold = useStore((state) => state.threshold);
  const [data, setData] = useState<Case | null>(null);
  const [error, setError] = useState<string | null>(null);
  const panel = useRef<HTMLElement>(null);

  useEffect(() => {
    if (selected === null) return;
    let cancelled = false;
    setError(null);
    fetchCase(selected)
      .then((found) => !cancelled && setData(found))
      .catch((reason: Error) => !cancelled && (setData(null), setError(reason.message)));
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const open = selected !== null;
  useEffect(() => {
    if (!open) return;
    // Return focus to whatever opened the drawer when it closes.
    const opener = document.activeElement as HTMLElement | null;
    panel.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      // The How-it-works dialog handles its own Escape.
      if (event.key === 'Escape' && !document.querySelector('dialog[open]')) store.select(null);
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      if (opener?.isConnected) opener.focus({ preventScroll: true });
    };
  }, [open]);

  if (selected === null) return null;
  const loading = !error && (!data || unpack(data.transaction).id !== selected);

  return (
    <aside className="drawer" ref={panel} tabIndex={-1} aria-label="Payment details">
      <button type="button" className="drawer__close" onClick={() => store.select(null)} aria-label="Close">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6 6l12 12M18 6 6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </button>
      {error && <p className="drawer__error">{error}</p>}
      {loading && !error && <p className="muted drawer__loading">Loading…</p>}
      {!loading && data && <CaseBody data={data} threshold={threshold} />}
    </aside>
  );
}

function CaseBody({ data, threshold }: { data: Case; threshold: number }) {
  const txn = unpack(data.transaction);
  const declined = txn.score > threshold;
  const attack = txn.attack ? ATTACKS[txn.attack - 1] : undefined;
  const verdict = declined ? (txn.fraud ? 'Caught' : 'False alarm') : txn.fraud ? 'Missed' : 'Correct';

  return (
    <div className="case">
      <header className="case__head">
        <p className="eyebrow">
          Payment #{txn.id} · {clock(txn.ts)}
        </p>
        <h2 className="case__amount">{money(txn.amount)}</h2>
        <p className="case__where">
          {data.merchant} · {CATEGORY_LABELS[txn.category]} · {data.online ? 'online, from' : 'in store,'} {data.city}
          {data.country !== 'US' ? `, ${data.country}` : ''}
        </p>
        <div className="case__decision">
          <span className={`pill ${declined ? 'pill--declined' : 'pill--approved'}`}>{declined ? 'Declined' : 'Approved'}</span>
          <span className={`pill pill--quiet verdict--${verdict.replace(' ', '-').toLowerCase()}`}>
            {verdict}: {attack ? `actually ${attack.label.toLowerCase()}` : 'actually an honest customer'}
          </span>
        </div>
      </header>

      <Gauge score={txn.score} threshold={threshold} latency={data.latency_us} />

      <section className="case__section">
        <h3 className="subhead">What moved the score</h3>
        <Reasons risk={data.explanation.risk} reassuring={data.explanation.reassuring} />
      </section>

      {data.ring && data.ring.accounts.length >= 3 && (
        <section className="case__section">
          <h3 className="subhead">Linked accounts</h3>
          <p className="muted small">
            {data.ring.accounts.length} accounts share {data.ring.devices.length} {data.ring.devices.length === 1 ? 'device' : 'devices'} and {data.ring.addresses.length}{' '}
            {data.ring.addresses.length === 1 ? 'address' : 'addresses'}.
            {data.ring.accounts.length <= 4 && data.customer.synthetic === false ? ' A small group like this is usually a household.' : ''}
            {data.ring.new_accounts.length ? ` ${data.ring.new_accounts.length} were opened in the last two weeks.` : ''}
          </p>
          <RingGraph ring={data.ring} account={data.customer.id} />
        </section>
      )}

      {data.related.length > 1 && data.attack && (
        <section className="case__section">
          <h3 className="subhead">This {data.attack.label.toLowerCase()} attack</h3>
          <Timeline txns={data.related.map(unpack)} current={txn.id} threshold={threshold} />
        </section>
      )}

      <section className="case__section">
        <h3 className="subhead">The cardholder</h3>
        <dl className="facts">
          <dt>Card</dt>
          <dd>•••• {data.customer.last4}</dd>
          <dt>Lives in</dt>
          <dd>{data.customer.home}</dd>
          <dt>Account age</dt>
          <dd>{data.customer.account_days < 1 ? 'Opened today' : `${Math.round(data.customer.account_days).toLocaleString()} days`}</dd>
          {data.customer.synthetic && (
            <>
              <dt>Identity</dt>
              <dd className="danger">Fake, made by a fraud ring</dd>
            </>
          )}
        </dl>
        {data.history.length > 0 && (
          <>
            <h4 className="microhead">Earlier on this card</h4>
            <ul className="history">
              {data.history
                .map(unpack)
                .reverse()
                .map((item) => (
                  <li key={item.id}>
                    <button type="button" onClick={() => store.select(item.id)}>
                      <span>{clock(item.ts)}</span>
                      <span>{item.merchant}</span>
                      <span>{money(item.amount)}</span>
                      <span className={item.score > threshold ? 'danger' : 'muted'}>{item.score > threshold ? 'Declined' : 'Approved'}</span>
                    </button>
                  </li>
                ))}
            </ul>
          </>
        )}
      </section>
    </div>
  );
}

function Gauge({ score, threshold, latency }: { score: number; threshold: number; latency: number }) {
  // Log-odds scale so 0.1% and 1% are visibly different.
  const position = (p: number) => {
    const clamped = Math.min(Math.max(p, 1e-4), 1 - 1e-4);
    return ((Math.log(clamped / (1 - clamped)) + 9.2) / 18.4) * 100;
  };
  return (
    <div className="gauge">
      <div className="gauge__top">
        <span>
          Risk <b>{score < 0.001 ? '<0.1' : (score * 100).toFixed(score < 0.1 ? 1 : 0)}</b>
          <span className="muted"> / 100</span>
        </span>
        <span className="muted small">scored in {microseconds(latency)}</span>
      </div>
      <div className="gauge__track">
        <span className="gauge__declined" style={{ left: `${position(threshold)}%` }} />
        <span className="gauge__marker" style={{ left: `${position(score)}%` }} />
      </div>
      <div className="gauge__ends muted small">
        <span>Safe</span>
        <span style={{ left: `${position(threshold)}%` }} className="gauge__threshold">
          Decline line
        </span>
        <span>Fraud</span>
      </div>
    </div>
  );
}

function Reasons({ risk, reassuring }: { risk: Reason[]; reassuring: Reason[] }) {
  const all = [...risk, ...reassuring];
  if (!all.length) return <p className="muted">Nothing stood out. This looks like the customer's normal spending.</p>;
  const largest = Math.max(...all.map((reason) => Math.abs(reason.impact)));
  return (
    <>
      <ul className="reasons">
        {all.map((reason) => (
          <li key={reason.feature} className={reason.impact > 0 ? 'is-risk' : 'is-safe'}>
            <span className="reasons__text">{reason.text}</span>
            <span className="reasons__bar" aria-hidden="true">
              <span style={{ width: `${(Math.abs(reason.impact) / largest) * 100}%` }} />
            </span>
            <span className="reasons__impact">
              {reason.impact > 0 ? '+' : '−'}
              {Math.abs(reason.impact).toFixed(1)}
            </span>
          </li>
        ))}
      </ul>
      <p className="muted small">Each signal's push on the risk score, from the model's own SHAP values.</p>
    </>
  );
}

function Timeline({ txns, current, threshold }: { txns: Txn[]; current: number; threshold: number }) {
  const firstBlocked = txns.findIndex((txn) => txn.score > threshold);
  const blocked = txns.filter((txn) => txn.score > threshold).length;
  return (
    <>
      <ol className="timeline" aria-label="Payments in this attack, in order">
        {txns.map((txn, index) => (
          <li key={txn.id}>
            <button
              type="button"
              className={`timeline__tick${txn.score > threshold ? ' is-declined' : ''}${txn.id === current ? ' is-current' : ''}`}
              style={{ height: `${18 + Math.min(1, Math.log10(txn.amount + 1) / 3.5) * 30}px` }}
              onClick={() => store.select(txn.id)}
              aria-label={`Payment ${index + 1}: ${money(txn.amount)}, ${txn.score > threshold ? 'declined' : 'approved'}`}
              title={`${money(txn.amount)} at ${txn.merchant}`}
            />
          </li>
        ))}
      </ol>
      <p className="muted small">
        {blocked} of {txns.length} payments blocked
        {firstBlocked >= 0 ? `, starting with payment ${firstBlocked + 1}` : ''}. Bar height is the amount.
      </p>
    </>
  );
}
