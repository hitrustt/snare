import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, type SimulationLinkDatum, type SimulationNodeDatum } from 'd3-force';
import { useMemo } from 'react';
import type { Case } from '../api.ts';

interface Node extends SimulationNodeDatum {
  id: string;
  kind: 'a' | 'd' | 's';
  flagged: boolean;
  fresh: boolean;
  self: boolean;
}

const W = 360;
const H = 240;

/**
 * The accounts linked to this one through shared phones and shipping
 * addresses. A single account looks normal; forty accounts created last week
 * that all log in from the same three phones do not.
 */
export function RingGraph({ ring, account }: { ring: NonNullable<Case['ring']>; account: number }) {
  const layout = useMemo(() => {
    const flagged = new Set(ring.flagged_accounts);
    const fresh = new Set(ring.new_accounts);
    const nodes: Node[] = [
      ...ring.accounts.map((id) => ({ id: `a${id}`, kind: 'a' as const, flagged: flagged.has(id), fresh: fresh.has(id), self: id === account })),
      ...ring.devices.map((id) => ({ id: `d${id}`, kind: 'd' as const, flagged: false, fresh: false, self: false })),
      ...ring.addresses.map((id) => ({ id: `s${id}`, kind: 's' as const, flagged: false, fresh: false, self: false })),
    ];
    const ids = new Set(nodes.map((node) => node.id));
    const links: SimulationLinkDatum<Node>[] = ring.edges.filter(([a, b]) => ids.has(a) && ids.has(b)).map(([source, target]) => ({ source, target }));
    const simulation = forceSimulation(nodes)
      .force('link', forceLink<Node, SimulationLinkDatum<Node>>(links).id((node) => node.id).distance(26).strength(0.9))
      .force('charge', forceManyBody().strength(-46))
      .force('collide', forceCollide(9))
      .force('center', forceCenter(W / 2, H / 2))
      .stop();
    for (let i = 0; i < 260; i += 1) simulation.tick();

    // Fit to the box.
    const xs = nodes.map((node) => node.x ?? 0);
    const ys = nodes.map((node) => node.y ?? 0);
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const scale = Math.min((W - 28) / Math.max(maxX - minX, 1), (H - 28) / Math.max(maxY - minY, 1), 1.6);
    const fit = (node: Node) => ({ x: W / 2 + ((node.x ?? 0) - (minX + maxX) / 2) * scale, y: H / 2 + ((node.y ?? 0) - (minY + maxY) / 2) * scale });
    const placed = new Map(nodes.map((node) => [node.id, { ...node, ...fit(node) }]));
    return {
      nodes: [...placed.values()],
      links: links.map((link) => ({ a: placed.get((link.source as Node).id)!, b: placed.get((link.target as Node).id)! })),
    };
  }, [ring, account]);

  return (
    <figure className="ring">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${ring.accounts.length} accounts linked by ${ring.devices.length} shared devices and ${ring.addresses.length} shared addresses`}>
        {layout.links.map((link, index) => (
          <line key={index} x1={link.a.x} y1={link.a.y} x2={link.b.x} y2={link.b.y} className="ring__edge" />
        ))}
        {layout.nodes.map((node) =>
          node.kind === 'a' ? (
            <circle key={node.id} cx={node.x} cy={node.y} r={node.self ? 7 : 5} className={`ring__account${node.flagged ? ' is-flagged' : ''}${node.fresh ? ' is-new' : ''}${node.self ? ' is-self' : ''}`} />
          ) : node.kind === 'd' ? (
            <rect key={node.id} x={node.x - 4.5} y={node.y - 6} width={9} height={12} rx={2} className="ring__device" />
          ) : (
            <path key={node.id} d={`M${node.x - 5},${node.y + 4}V${node.y - 1}L${node.x},${node.y - 6}L${node.x + 5},${node.y - 1}V${node.y + 4}Z`} className="ring__address" />
          ),
        )}
      </svg>
      <figcaption className="ring__legend">
        <span className="key key--account">Account</span>
        <span className="key key--new">Opened in the last 2 weeks</span>
        <span className="key key--flagged">Declined recently</span>
        <span className="key key--device">Phone or computer</span>
        <span className="key key--address">Shipping address</span>
      </figcaption>
    </figure>
  );
}
