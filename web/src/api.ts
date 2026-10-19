/**
 * Talking to the Python server: a WebSocket of scored payments, plus a few
 * JSON endpoints. Payments arrive as compact arrays (see snare/engine.py
 * `compact`) and are unpacked here into objects once.
 */

export interface Txn {
  id: number;
  ts: number; // simulated seconds
  amount: number;
  category: number;
  online: boolean;
  lat: number;
  lng: number;
  city: number;
  score: number; // 0..1, model probability of fraud
  fraud: boolean; // ground truth (known because the network is simulated)
  attack: number; // 0 none, else index into ATTACKS + 1
  scenario: number;
  merchant: string;
  last4: string;
}

export type WireTxn = [number, number, number, number, number, number, number, number, number, number, number, number, string, string];

export function unpack(row: WireTxn): Txn {
  return {
    id: row[0],
    ts: row[1],
    amount: row[2],
    category: row[3],
    online: row[4] === 1,
    lat: row[5],
    lng: row[6],
    city: row[7],
    score: row[8],
    fraud: row[9] === 1,
    attack: row[10],
    scenario: row[11],
    merchant: row[12],
    last4: row[13],
  };
}

export interface Attack {
  scenario: number;
  kind: AttackKind;
  label: string;
  source: 'visitor' | 'background';
  start: number;
  end: number;
  transactions: number;
  accounts: number;
  seen: number;
  caught: number;
  dollars: number;
  dollars_caught: number;
}

export interface Stats {
  transactions: number;
  declined: number;
  fraud: number;
  fraud_caught: number;
  false_declines: number;
  fraud_dollars: number;
  fraud_dollars_caught: number;
  p50_us: number;
  p99_us: number;
}

export type AttackKind = 'card_testing' | 'account_takeover' | 'skimmed_card' | 'fraud_ring';

export const ATTACKS: { kind: AttackKind; label: string; summary: string }[] = [
  { kind: 'card_testing', label: 'Card testing', summary: 'Tiny test charges on a stolen card, then a cash-out.' },
  { kind: 'account_takeover', label: 'Account takeover', summary: 'A login from a new device, then big purchases, fast.' },
  { kind: 'skimmed_card', label: 'Cloned card', summary: 'A copied card used in stores the owner never visits.' },
  { kind: 'fraud_ring', label: 'Fraud ring', summary: 'Fake accounts on a few shared phones, each buying a little.' },
];

export const CATEGORY_LABELS = [
  'Groceries', 'Restaurant', 'Gas station', 'Retail', 'Online store', 'Rideshare', 'Pharmacy',
  'Subscription', 'Apps and games', 'Electronics', 'Travel', 'Home goods', 'Gift cards',
];

export interface Reason {
  feature: string;
  text: string;
  impact: number;
}

export interface Case {
  transaction: WireTxn;
  merchant: string;
  category: string;
  online: boolean;
  city: string;
  country: string;
  customer: { id: number; last4: string; home: string; account_days: number; synthetic: boolean };
  score: number;
  threshold: number;
  latency_us: number;
  explanation: { base_log_odds: number; risk: Reason[]; reassuring: Reason[] };
  ring: null | {
    accounts: number[];
    devices: number[];
    addresses: number[];
    edges: [string, string][];
    new_accounts: number[];
    flagged_accounts: number[];
  };
  attack: Attack | null;
  related: WireTxn[];
  history: WireTxn[];
}

export interface ModelReport {
  recall: number;
  dollar_recall: number;
  attacks_caught: number;
  precision: number;
  false_positive_rate: number;
  pr_auc: number;
  roc_auc: number;
  transactions: number;
  fraud_transactions: number;
  recall_by_type: Record<string, number>;
}

export interface ModelInfo {
  metrics: {
    dataset: { customers: number; days: number; transactions: number; fraud_transactions: number };
    operating_point: { target_false_positive_rate: number; threshold: number };
    model: ModelReport;
    rules_baseline: ModelReport;
    without_graph_features: ModelReport;
    feature_importance_gain: Record<string, number>;
  };
  threshold: number;
  speed: number;
  customers: number;
  cities: { name: string; country: string; lat: number; lng: number }[];
}

export type StreamMessage =
  | { type: 'hello'; now: number; threshold: number; txns: WireTxn[]; attacks: Attack[]; stats: Stats }
  | { type: 'tick'; now: number; txns: WireTxn[]; events: ({ type: 'attack' } & Attack)[]; attacks: Attack[]; stats: Stats };

export async function launchAttack(kind: AttackKind): Promise<Attack> {
  const response = await fetch('/api/attacks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? body.detail ?? 'The attack could not be launched.');
  return body as Attack;
}

export async function fetchCase(id: number): Promise<Case> {
  const response = await fetch(`/api/cases/${id}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? 'That payment could not be loaded.');
  }
  return response.json();
}

export async function fetchModel(): Promise<ModelInfo> {
  const response = await fetch('/api/model');
  if (!response.ok) throw new Error('Model details are unavailable.');
  return response.json();
}

/** Reconnecting WebSocket. Free hosts sleep when idle, so the first connect can take a while. */
export function connect(onMessage: (message: StreamMessage) => void, onStatus: (status: 'connecting' | 'live' | 'reconnecting') => void) {
  let socket: WebSocket | null = null;
  let attempts = 0;
  let connected = false;
  let closed = false;
  let ping: ReturnType<typeof setInterval> | undefined;
  let retry: ReturnType<typeof setTimeout> | undefined;

  const open = () => {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${scheme}://${location.host}/api/stream`);
    socket.onopen = () => {
      attempts = 0;
      ping = setInterval(() => socket?.readyState === WebSocket.OPEN && socket.send('ping'), 20_000);
    };
    // Live only once data flows: a server that is still warming up accepts
    // the connection but sends nothing until it is ready.
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data as string) as StreamMessage;
      if (message.type === 'hello') {
        connected = true;
        onStatus('live');
      }
      onMessage(message);
    };
    socket.onclose = () => {
      clearInterval(ping);
      if (closed) return;
      onStatus(connected ? 'reconnecting' : 'connecting');
      attempts += 1;
      retry = setTimeout(open, Math.min(1000 * 2 ** attempts, 15_000));
    };
  };

  onStatus('connecting');
  open();
  return () => {
    closed = true;
    clearInterval(ping);
    clearTimeout(retry);
    socket?.close();
  };
}
