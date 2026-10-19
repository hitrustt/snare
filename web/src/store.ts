/**
 * Client state: a rolling buffer of recent payments plus the server's
 * counters. Decisions are recomputed in the browser from each payment's
 * score, so moving the strictness slider re-judges everything instantly
 * without asking the server.
 */

import { useSyncExternalStore } from 'react';
import { unpack, type Attack, type Stats, type StreamMessage, type Txn } from './api.ts';

const BUFFER = 18_000; // a little less than the server keeps, so every payment shown can be opened

export interface State {
  status: 'connecting' | 'live' | 'reconnecting';
  now: number;
  txns: Txn[]; // oldest first, capped at BUFFER
  attacks: Attack[];
  stats: Stats | null;
  serverThreshold: number;
  threshold: number;
  version: number;
  selected: number | null;
  notice: string | null;
}

let state: State = {
  status: 'connecting',
  now: 0,
  txns: [],
  attacks: [],
  stats: null,
  serverThreshold: 0.5,
  threshold: 0.5,
  version: 0,
  selected: null,
  notice: null,
};
const listeners = new Set<() => void>();

function set(next: Partial<State>) {
  state = { ...state, ...next };
  listeners.forEach((listener) => listener());
}

export const store = {
  get: () => state,
  subscribe(listener: () => void) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  setStatus: (status: State['status']) => set({ status }),
  setThreshold: (threshold: number) => set({ threshold, version: state.version + 1 }),
  select: (selected: number | null) => set({ selected }),
  notify(notice: string | null) {
    set({ notice });
  },
  receive(message: StreamMessage) {
    const incoming = message.txns.map(unpack);
    if (message.type === 'hello') {
      const threshold = state.version === 0 ? message.threshold : state.threshold;
      set({ now: message.now, txns: incoming, attacks: message.attacks, stats: message.stats, serverThreshold: message.threshold, threshold, version: state.version + 1 });
      return;
    }
    const txns = state.txns.length + incoming.length > BUFFER ? [...state.txns.slice(state.txns.length + incoming.length - BUFFER), ...incoming] : [...state.txns, ...incoming];
    set({ now: message.now, txns, attacks: message.attacks, stats: message.stats, version: state.version + 1 });
  },
};

export function useStore<T>(selector: (state: State) => T): T {
  return useSyncExternalStore(store.subscribe, () => selector(state), () => selector(state));
}

export const declined = (txn: Txn, threshold: number) => txn.score > threshold;
