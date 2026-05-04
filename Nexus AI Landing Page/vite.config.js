import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/proxy/18300/',
  server: {
    host: '127.0.0.1',
    port: 18300,
    hmr: false, // HMR WebSocket cannot pass through the reverse proxy
  },
});

 // HMR WebSocket cannot pass through the reverse proxy

