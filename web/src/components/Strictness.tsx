import { useMemo } from 'react';
import { store, useStore } from '../store.ts';
import { count, oneIn, percent } from '../format.ts';

const LOW = -5; // logit(0.0067)
const HIGH = 5; // logit(0.9933)
const toLogit = (p: number) => Math.log(p / (1 - p));
const fromLogit = (x: number) => 1 / (1 + Math.exp(-x));

/**
 * Every fraud system trades blocked fraud against blocked customers. This
 * slider moves the decision threshold and re-judges every payment on screen,
 * so the trade-off is something you can feel rather than read about.
 */
export function Strictness() {
  const threshold = useStore((state) => state.threshold);
  const serverThreshold = useStore((state) => state.serverThreshold);
  const txns = useStore((state) => state.txns);

  const curve = useMemo(() => {
    const fraud: number[] = [];
    const honest: number[] = [];
    for (const txn of txns) (txn.fraud ? fraud : honest).push(txn.score);
    fraud.sort((a, b) => a - b);
    honest.sort((a, b) => a - b);
    const above = (sorted: number[], t: number) => {
      let lo = 0;
      let hi = sorted.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if ((sorted[mid] as number) > t) hi = mid;
        else lo = mid + 1;
      }
      return sorted.length - lo;
    };
    const at = (t: number) => ({
      caught: fraud.length ? above(fraud, t) / fraud.length : 0,
      declined: honest.length ? above(honest, t) / honest.length : 0,
      caughtCount: above(fraud, t),
      declinedCount: above(honest, t),
    });
    const points = [];
    for (let step = 0; step <= 60; step += 1) {
      const t = fromLogit(HIGH - ((HIGH - LOW) * step) / 60);
      points.push({ t, ...at(t) });
    }
    return { at, points, fraud: fraud.length, honest: honest.length };
  }, [txns]);

  const now = curve.at(threshold);
  // 0 = lenient (high threshold), 1 = strict (low threshold).
  const position = (HIGH - toLogit(threshold)) / (HIGH - LOW);
  const trained = (HIGH - toLogit(serverThreshold)) / (HIGH - LOW);

  // Chart: x is strictness, two lines: share of fraud caught, honest customers declined (log scale).
  const W = 280;
  const H = 96;
  const x = (step: number) => (step / 60) * W;
  const yCaught = (v: number) => H - 4 - v * (H - 8);
  const yDeclined = (v: number) => {
    const clamped = Math.max(v, 1e-4);
    return H - 4 - ((Math.log10(clamped) + 4) / 4) * (H - 8);
  };
  const caughtPath = curve.points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${yCaught(p.caught).toFixed(1)}`).join('');
  const declinedPath = curve.points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${yDeclined(p.declined).toFixed(1)}`).join('');

  return (
    <section className="panel strict" aria-labelledby="strict-title">
      <div className="strict__head">
        <h2 id="strict-title" className="panel-title">
          Strictness
        </h2>
        {Math.abs(threshold - serverThreshold) > 1e-6 && (
          <button type="button" className="link" onClick={() => store.setThreshold(serverThreshold)}>
            Reset to trained
          </button>
        )}
      </div>

      <div className="strict__readout">
        <div>
          <span className="strict__big">{curve.fraud ? percent(now.caught, 0) : '–'}</span>
          <span className="strict__label">of fraud blocked</span>
        </div>
        <div>
          <span className="strict__big strict__big--honest">{curve.honest ? oneIn(now.declined) : '–'}</span>
          <span className="strict__label">honest payments declined</span>
        </div>
      </div>

      <svg className="strict__chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
        <line x1={trained * W} x2={trained * W} y1={0} y2={H} className="strict__trained" />
        <path d={declinedPath} className="strict__declined" />
        <path d={caughtPath} className="strict__caught" />
        <line x1={position * W} x2={position * W} y1={0} y2={H} className="strict__cursor" />
      </svg>
      <div className="strict__legend" aria-hidden="true">
        <span className="key key--caught">Fraud blocked</span>
        <span className="key key--declined">Honest declined (log)</span>
        <span className="key key--trained">Trained setting</span>
      </div>

      <label className="strict__slider">
        <span className="visually-hidden">Decline payments with a risk score above</span>
        <input
          type="range"
          min={0}
          max={1000}
          value={Math.round(position * 1000)}
          onChange={(event) => store.setThreshold(fromLogit(HIGH - ((HIGH - LOW) * Number(event.target.value)) / 1000))}
          aria-valuetext={`Decline above risk ${Math.round(threshold * 100)} of 100`}
        />
        <span className="strict__ends">
          <span>Lenient</span>
          <span>Decline above risk {(threshold * 100).toFixed(threshold < 0.1 ? 1 : 0)}</span>
          <span>Strict</span>
        </span>
      </label>
      <p className="muted small">
        Re-judges the last {count(curve.fraud + curve.honest)} payments in your browser. {curve.fraud ? `${count(now.caughtCount)} of ${count(curve.fraud)} fraudulent, ${count(now.declinedCount)} honest declined.` : ''}
      </p>
    </section>
  );
}
