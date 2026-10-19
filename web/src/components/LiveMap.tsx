import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import { useEffect, useRef } from 'react';
import type { Txn } from '../api.ts';
import { store } from '../store.ts';

maplibregl.setWorkerUrl(workerUrl);

/** How long a payment stays on the map, in simulated seconds. */
const TRAIL_SECONDS = 3 * 60 * 60;
const FLASH_SECONDS = 6 * 60;

/**
 * Every payment lands on the map where it happened (the store, or the
 * network location for online purchases). Approved payments are small ink
 * dots that fade out; declined ones flash red. Fraud the model let through
 * is drawn as an amber ring, so misses are as visible as catches.
 */
export function LiveMap({ onSelect }: { onSelect: (id: number) => void }) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const ready = useRef(false);

  useEffect(() => {
    if (!container.current) return;
    const instance = new maplibregl.Map({
      container: container.current,
      style: 'https://tiles.openfreemap.org/styles/positron',
      bounds: [
        [-124, 25],
        [-67, 49],
      ],
      fitBoundsOptions: { padding: 24 },
      minZoom: 1.5,
      maxZoom: 9,
      attributionControl: { compact: true },
      dragRotate: false,
      pitchWithRotate: false,
    });
    instance.touchZoomRotate.disableRotation();
    map.current = instance;

    instance.on('load', () => {
      instance.addSource('payments', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      instance.addLayer({
        id: 'approved',
        type: 'circle',
        source: 'payments',
        filter: ['==', ['get', 'kind'], 'approved'],
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 3, 2, 8, 4],
          'circle-color': '#2b3440',
          'circle-opacity': ['+', 0.08, ['*', 0.5, ['get', 'fade']]],
        },
      });
      instance.addLayer({
        id: 'missed',
        type: 'circle',
        source: 'payments',
        filter: ['==', ['get', 'kind'], 'missed'],
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 3, 4, 8, 7],
          'circle-color': 'rgba(0,0,0,0)',
          'circle-stroke-color': '#c27c0e',
          'circle-stroke-width': 2,
          'circle-stroke-opacity': ['get', 'fade'],
        },
      });
      instance.addLayer({
        id: 'declined-halo',
        type: 'circle',
        source: 'payments',
        filter: ['all', ['==', ['get', 'kind'], 'declined'], ['>', ['get', 'flash'], 0]],
        paint: {
          'circle-radius': ['+', 6, ['*', 18, ['-', 1, ['get', 'flash']]]],
          'circle-color': '#e5484d',
          'circle-opacity': ['*', 0.35, ['get', 'flash']],
        },
      });
      instance.addLayer({
        id: 'declined',
        type: 'circle',
        source: 'payments',
        filter: ['==', ['get', 'kind'], 'declined'],
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 3, 3.6, 8, 6],
          'circle-color': '#d92d20',
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 1.2,
          'circle-opacity': ['max', 0.35, ['get', 'fade']],
        },
      });
      ready.current = true;
      draw();
    });

    for (const layer of ['declined', 'missed']) {
      instance.on('click', layer, (event) => {
        const id = event.features?.[0]?.properties?.id;
        if (typeof id === 'number') onSelect(id);
      });
      instance.on('mouseenter', layer, () => (instance.getCanvas().style.cursor = 'pointer'));
      instance.on('mouseleave', layer, () => (instance.getCanvas().style.cursor = ''));
    }

    const draw = () => {
      if (!ready.current) return;
      const { txns, now, threshold } = store.get();
      const features = [];
      for (let index = txns.length - 1; index >= 0; index -= 1) {
        const txn = txns[index] as Txn;
        const age = now - txn.ts;
        if (age > TRAIL_SECONDS) break;
        const isDeclined = txn.score > threshold;
        const kind = isDeclined ? 'declined' : txn.fraud ? 'missed' : 'approved';
        features.push({
          type: 'Feature' as const,
          properties: { id: txn.id, kind, fade: Math.max(0, 1 - age / TRAIL_SECONDS), flash: isDeclined ? Math.max(0, 1 - age / FLASH_SECONDS) : 0 },
          geometry: { type: 'Point' as const, coordinates: [txn.lng, txn.lat] },
        });
      }
      (instance.getSource('payments') as maplibregl.GeoJSONSource | undefined)?.setData({ type: 'FeatureCollection', features });
    };

    const unsubscribe = store.subscribe(draw);
    const resize = new ResizeObserver(() => instance.resize());
    resize.observe(container.current);
    return () => {
      unsubscribe();
      resize.disconnect();
      instance.remove();
    };
  }, [onSelect]);

  return <div ref={container} className="map" role="img" aria-label="Map of recent payments; declined payments are shown in red" />;
}
