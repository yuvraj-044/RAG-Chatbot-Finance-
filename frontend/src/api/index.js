// ── API Router ──────────────────────────────────────────────────────────
// If VITE_API_URL is set → use real backend.
// Otherwise → use mock API for standalone dev/demo.
//
// TO SWITCH TO REAL BACKEND:
//   1. Create frontend/.env with:  VITE_API_URL=https://your-space.hf.space
//   2. Restart the dev server (npm run dev)
//   That's it — no code changes needed.

import { mockChat, mockHealth, mockReset } from "./mock.js";
import { apiChat, apiHealth, apiReset } from "./client.js";

const USE_REAL_API = Boolean(import.meta.env.VITE_API_URL);

export const chat = USE_REAL_API ? apiChat : mockChat;
export const health = USE_REAL_API ? apiHealth : mockHealth;
export const reset = USE_REAL_API ? apiReset : mockReset;

export { USE_REAL_API };
