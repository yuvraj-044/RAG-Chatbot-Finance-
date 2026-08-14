# Manual Test Checklist — FinSight Frontend

> Run this after wiring to the real backend (`VITE_API_URL` set).
> All tests should also pass in mock mode (no `VITE_API_URL`).

---

## 1. Normal Query ✅

- [ ] Open `http://localhost:5173` — landing page loads with starter chips
- [ ] Click "What was Q3 revenue for AAPL?" starter chip
- [ ] Verify: user message appears right-aligned in green bubble
- [ ] Verify: loading skeleton appears while waiting
- [ ] Verify: bot response appears with typewriter animation
- [ ] Verify: latency badge shows (e.g., "⚡ 142ms")
- [ ] Verify: confidence badge shows (e.g., "✓ 92%")
- [ ] Verify: grounded badge shows "✓ Grounded"
- [ ] Click "View Sources (2)" chip
- [ ] Verify: source drawer slides in from right
- [ ] Verify: each source card shows doc title, chunk text, score bar, date
- [ ] Close drawer (click X or overlay or press Escape)
- [ ] Type a follow-up question manually and press Enter

## 2. Out-of-Scope Query 🔍

- [ ] Type "What's the weather today?" and send
- [ ] Verify: bot responds with a helpful "no relevant data" message
- [ ] Verify: no sources chip (or sources count = 0)
- [ ] Verify: "⚠ Unverified" badge appears (is_grounded = false)
- [ ] Verify: empty state icon appears below the message

## 3. Slow / Timeout Scenario ⏱️

- [ ] *(Mock mode)* Observe occasional 1-2s delays — skeleton stays visible throughout
- [ ] *(Real backend)* If backend takes >30s, verify:
  - Error banner slides down from top
  - Banner shows "Request timed out" message
  - "Retry" button is visible and functional
  - Banner auto-dismisses after 8 seconds

## 4. Malformed Response / API Error 💥

- [ ] *(To test)* Temporarily set `VITE_API_URL` to a bad URL (e.g., `http://localhost:9999`)
- [ ] Send a message
- [ ] Verify: error banner appears with user-friendly message
- [ ] Verify: no blank screen, no console stack trace shown to user
- [ ] Verify: retry button works (still fails but shows banner again)
- [ ] Reset `VITE_API_URL` and verify normal operation resumes

## 5. Theme Toggle 🌙☀️

- [ ] Click theme toggle (moon icon) — switches to light mode
- [ ] Verify: all elements update (header, bubbles, input, drawer)
- [ ] Verify: no flash of unstyled content
- [ ] Refresh page — theme persists (stored in localStorage)
- [ ] Toggle back to dark mode

## 6. New Session 🗑️

- [ ] After sending a few messages, click "New Chat" button
- [ ] Verify: all messages cleared
- [ ] Verify: starter chips reappear
- [ ] Verify: new messages go to a fresh session

## 7. Mobile Responsive 📱

- [ ] Open in Chrome DevTools → Toggle device toolbar (375px width)
- [ ] Verify: header condensed, subtitle hidden
- [ ] Verify: starter chips stack to single column
- [ ] Verify: messages fill width (95%)
- [ ] Verify: source drawer becomes full-screen overlay
- [ ] Verify: input bar is accessible and not cut off

## 8. Backend Health Indicator 🟢

- [ ] With mock mode: header shows "Demo Mode" with green dot
- [ ] With real backend: header shows "API Connected" with green dot
- [ ] With no backend: header shows "Offline" with red dot
