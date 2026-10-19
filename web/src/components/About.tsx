import { useEffect, useRef, useState } from 'react';
import { fetchModel, type ModelInfo, type ModelReport } from '../api.ts';
import { count, oneIn, percent } from '../format.ts';

const TYPE_LABELS: Record<string, string> = {
  card_testing: 'Card testing',
  account_takeover: 'Account takeover',
  skimmed_card: 'Cloned card',
  fraud_ring: 'Fraud ring',
};

const FEATURE_LABELS: Record<string, string> = {
  amount_log: 'Amount',
  amount_vs_card: 'Amount vs. this card’s usual',
  amount_vs_category: 'Amount vs. the category’s usual',
  card_count_2m: 'Payments in the last 2 minutes',
  card_count_10m: 'Payments in the last 10 minutes',
  card_count_1h: 'Payments in the last hour',
  card_count_24h: 'Payments in the last day',
  card_sum_1h_log: 'Spent in the last hour',
  card_small_10m: 'Tiny payments in 10 minutes',
  card_merchants_1h: 'Different merchants in an hour',
  secs_since_last_log: 'Time since the last payment',
  km_from_home_log: 'Distance from home',
  km_from_last_log: 'Distance from the last payment',
  speed_from_last_kmh_log: 'Travel speed implied',
  store_speed_kmh_log: 'Speed between store visits',
  foreign: 'Abroad',
  new_merchant: 'New merchant',
  new_category: 'New kind of purchase',
  category_share: 'How often they buy this',
  device_known_min_log: 'How long the device is known',
  ip_known_min_log: 'How long the network is known',
  address_known_min_log: 'How long the address is known',
  hour_unusual: 'Unusual hour for them',
  account_age_days_log: 'Account age',
  card_history_log: 'Card history length',
  device_accounts: 'Other accounts on this device',
  address_accounts: 'Other accounts at this address',
  ip_cards_1h: 'Cards on this network in an hour',
  ring_size: 'Linked accounts',
  ring_new_accounts: 'New accounts among them',
  online: 'Online',
  category: 'Category',
  hour: 'Hour of day',
};

/** How it works, with the numbers from the held-out test set. */
export function About({ onClose }: { onClose: () => void }) {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    dialog.current?.showModal();
    fetchModel().then(setInfo, (reason: Error) => setError(reason.message));
  }, []);

  const metrics = info?.metrics;
  const importance = metrics ? Object.entries(metrics.feature_importance_gain).slice(0, 8) : [];
  const topGain = importance[0]?.[1] ?? 1;

  return (
    <dialog ref={dialog} className="about" onClose={onClose} onClick={(event) => event.target === dialog.current && dialog.current?.close()}>
      <div className="about__body">
        <button type="button" className="drawer__close" onClick={() => dialog.current?.close()} aria-label="Close">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6 6l12 12M18 6 6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </button>
        <p className="eyebrow">How it works</p>
        <h2 className="about__title">A payment network you can attack</h2>
        <p>
          A Python simulation runs {info ? count(info.customers) : 'thousands of'} cardholders who shop the way people do: groceries near home, subscriptions every month,
          trips, new phones, late-night food. Mixed in are four kinds of fraud, each written to look as much like honest behaviour as possible.
        </p>

        <ol className="pipeline">
          <li>
            <b>Stream features.</b> Each payment updates per-card, per-device and per-network state in a single pass: how fast the card is being used,
            how far it has moved, whether the device is new. Training and the live server use the same code, so nothing is computed with hindsight.
          </li>
          <li>
            <b>Account graph.</b> Accounts that share phones or shipping addresses are joined with union-find. A fraud ring is invisible one account
            at a time and obvious as a group.
          </li>
          <li>
            <b>Score.</b> A LightGBM model trained on {metrics ? count(metrics.dataset.transactions) : 'millions of'} simulated payments returns a risk score.
            The decline line is set so about {metrics ? oneIn(metrics.operating_point.target_false_positive_rate) : '1 in 200'} honest payments is declined.
          </li>
          <li>
            <b>Explain.</b> The model’s own SHAP values say which signals pushed each score up or down, turned into plain sentences.
          </li>
        </ol>

        {error && <p className="drawer__error">{error}</p>}
        {metrics && (
          <>
            <h3 className="subhead">Results on the last 15 days, never seen in training</h3>
            <div className="table-wrap">
              <table className="results">
                <thead>
                  <tr>
                    <th scope="col" />
                    <th scope="col">Snare</th>
                    <th scope="col">Without the graph</th>
                    <th scope="col">Hand-written rules</th>
                  </tr>
                </thead>
                <tbody>
                  <Row label="Fraudulent payments blocked" pick={(m) => percent(m.recall)} reports={metrics} />
                  <Row label="Fraud dollars blocked" pick={(m) => percent(m.dollar_recall)} reports={metrics} />
                  <Row label="Attacks stopped at least once" pick={(m) => percent(m.attacks_caught)} reports={metrics} />
                  <Row label="Declines that were fraud" pick={(m) => percent(m.precision)} reports={metrics} />
                  <Row label="Honest payments declined" pick={(m) => oneIn(m.false_positive_rate)} reports={metrics} />
                  <Row label="PR-AUC" pick={(m) => (m.pr_auc ? m.pr_auc.toFixed(3) : '–')} reports={metrics} />
                </tbody>
              </table>
            </div>

            <h3 className="subhead">Blocked, by kind of attack</h3>
            <ul className="bars">
              {Object.entries(metrics.model.recall_by_type).map(([kind, value]) => (
                <li key={kind}>
                  <span>{TYPE_LABELS[kind] ?? kind}</span>
                  <span className="bars__track">
                    <span style={{ width: `${value * 100}%` }} />
                  </span>
                  <span className="bars__value">{percent(value, 0)}</span>
                </li>
              ))}
            </ul>

            <h3 className="subhead">What the model relies on most</h3>
            <ul className="bars bars--quiet">
              {importance.map(([name, gain]) => (
                <li key={name}>
                  <span>{FEATURE_LABELS[name] ?? name}</span>
                  <span className="bars__track">
                    <span style={{ width: `${(gain / topGain) * 100}%` }} />
                  </span>
                  <span className="bars__value" />
                </li>
              ))}
            </ul>

            <p className="muted small">
              {count(metrics.dataset.customers)} customers over {metrics.dataset.days} days, {count(metrics.dataset.transactions)} payments of which{' '}
              {count(metrics.dataset.fraud_transactions)} were fraud. After a two-week warm-up, trained on days 14 to 70, tuned on days 70 to 85, tested on days 85 to 100. The
              rules column is a typical starter rulebook: velocity limits, a new device with a big amount, big purchases abroad, and impossible travel.
            </p>
          </>
        )}
      </div>
    </dialog>
  );
}

function Row({ label, pick, reports }: { label: string; pick: (report: ModelReport) => string; reports: ModelInfo['metrics'] }) {
  return (
    <tr>
      <th scope="row">{label}</th>
      <td className="is-ours">{pick(reports.model)}</td>
      <td>{pick(reports.without_graph_features)}</td>
      <td>{pick(reports.rules_baseline)}</td>
    </tr>
  );
}
