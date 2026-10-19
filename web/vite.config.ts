import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  worker: { format: 'es' },
  server: {
    // The Python server runs the simulation; the dev server proxies to it.
    proxy: { '/api': { target: 'http://localhost:8000', ws: true } },
  },
  build: {
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: { manualChunks: (id) => (id.includes('node_modules/maplibre-gl') ? 'maplibre' : undefined) },
    },
  },
});
