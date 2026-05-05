import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/proxy/40948/',
  server: {
    host: '127.0.0.1',
    port: 40948,
    hmr: false, // HMR WebSocket cannot pass through the reverse proxy
  },
});
