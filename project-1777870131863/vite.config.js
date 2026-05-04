import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// NOTE: This base and HMR clientPort are overridden at runtime by J-Agent's
// projectRunner.js, which injects the correct port before starting the server.
// This file is a safe fallback for local development outside of J-Agent.
export default defineConfig({
  plugins: [react()],
  server: {
    hmr: true,
  },
});
