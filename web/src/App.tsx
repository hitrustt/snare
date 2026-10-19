import { useCallback, useEffect, useState } from 'react';
import { connect, type Txn } from './api.ts';
import { store, useStore } from './store.ts';
import { About } from './components/About.tsx';
import { AttackConsole } from './components/AttackConsole.tsx';
import { CaseDrawer } from './components/CaseDrawer.tsx';
import { Header } from './components/Header.tsx';
import { LiveMap } from './components/LiveMap.tsx';
import { ReceiptTape } from './components/ReceiptTape.tsx';
import { Strictness } from './components/Strictness.tsx';

export function App() {
  const [about, setAbout] = useState(false);
  const status = useStore((state) => state.status);

  useEffect(() => connect(store.receive, store.setStatus), []);

  const select = useCallback((id: number) => store.select(id), []);

  // Jump to the most telling payment of an attack: the riskiest one.
  const focusAttack = useCallback((scenario: number) => {
    let best: Txn | undefined;
    for (const txn of store.get().txns) if (txn.scenario === scenario && (!best || txn.score > best.score)) best = txn;
    if (best) store.select(best.id);
    else store.notify('That attack has scrolled out of the recent history.');
  }, []);

  return (
    <div className="app">
      <Header onAbout={() => setAbout(true)} />
      <main className="layout">
        <div className="rail">
          <AttackConsole onFocusAttack={focusAttack} />
          <Strictness />
        </div>
        <div className="stage">
          <LiveMap onSelect={select} />
          <MapLegend />
          {status !== 'live' && <Waking status={status} />}
        </div>
        <ReceiptTape onSelect={select} />
      </main>
      <CaseDrawer />
      <Notice />
      {about && <About onClose={() => setAbout(false)} />}
    </div>
  );
}

function MapLegend() {
  return (
    <ul className="legend" aria-label="Map key">
      <li>
        <span className="dot dot--approved" /> Approved
      </li>
      <li>
        <span className="dot dot--declined" /> Declined
      </li>
      <li>
        <span className="dot dot--missed" /> Fraud that got through
      </li>
    </ul>
  );
}

function Waking({ status }: { status: string }) {
  return (
    <div className="waking" role="status">
      <span className="spinner" aria-hidden="true" />
      <p>
        <b>{status === 'connecting' ? 'Starting the payment network' : 'Connection lost, reconnecting'}</b>
        <span className="muted">
          {status === 'connecting' ? 'The free server sleeps when nobody is watching. Waking it takes up to a minute.' : 'Hang on a moment.'}
        </span>
      </p>
    </div>
  );
}

function Notice() {
  const notice = useStore((state) => state.notice);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => store.notify(null), 4500);
    return () => clearTimeout(timer);
  }, [notice]);
  return (
    <div className="notice-region" role="status" aria-live="polite">
      {notice && <div className="notice">{notice}</div>}
    </div>
  );
}
