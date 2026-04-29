# Project Context

Read this file at the start of every new chat to restore full project context.

---

## Project name

Pose Tracker (repo: `Gym_bro`)

## GitHub

- Remote: `https://github.com/AndrewLu-1/Gym_bro.git`
- Active branch: `main`
- `temp` branch holds the earlier, more complex version of the app

## What this project is

A browser-based real-time pose tracking app. It uses the device webcam and MediaPipe to detect body landmarks, draws a live skeleton overlay on the video feed, and exposes all joint positions as typed React state that updates every frame as the user moves.

This started as a fitness coaching tool (exercise form checker with rep counting), went through a major rewrite into a dual-canvas MoCap system, then was reset to a clean minimal pose tracker on `main`.

---

## Current state of `main`

### What works
- Webcam opens and streams to a canvas element
- Main branch is now a minimal browser pose tracker only — no squat logic, rep counting, or coaching features
- Tracking uses the legacy browser `@mediapipe/pose` pipeline instead of `@mediapipe/tasks-vision`
- Skeleton overlay draws on top of the live video — teal lines connecting joints, white/amber dots at each landmark
- `PoseData` React state updates from MediaPipe pose results with named joint objects (`poseData.leftKnee.x`, `poseData.rightWrist.screenX`, etc.)
- Right panel shows all 13 joint coordinates and visibility confidence live

### Known fixed bugs (do not reintroduce)
- **Stale hidden video element**: using `display: none` for the source video made the tracking path less reliable. Fixed by keeping the video mounted but visually hidden in `src/App.css`.
- **Detection before video ready**: `play()` resolving does not guarantee `videoWidth` is non-zero. Fixed by awaiting a `loadedmetadata` promise before starting the loop, plus explicit video readiness guards.
- **Pose visibility fallback**: missing `visibility` values caused weak UI/state behavior. Fixed to treat `visibility ?? 1` as visible.
- **Old `tasks-vision` pipeline removed**: the previous `@mediapipe/tasks-vision` setup was replaced entirely after repeated user reports that it was not detecting anything. Do not reintroduce it casually on `main`.
- **CDN asset version mismatch risk**: the new `@mediapipe/pose` loader now pins assets to the exact package version (`0.5.1675469404`) instead of using an unversioned CDN path.
- **Unhelpful model error text**: the generic "check your connection and refresh" message was replaced with real runtime error text from the loader. If that old message appears, the user is running a stale dev server or stale browser tab.

---

## Architecture

### Entry point
`src/main.tsx` → `src/App.tsx`

### Core hook: `src/hooks/usePoseTracking.ts`

This file contains everything. Understand this file and you understand the whole app.

**What it does:**
1. Loads the browser `@mediapipe/pose` solution on mount using `window.Pose`
2. Pins loader assets to `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/...`
3. Configures pose options:
   - `modelComplexity: 2`
   - `smoothLandmarks: true`
   - `enableSegmentation: false`
   - `minDetectionConfidence: 0.35`
   - `minTrackingConfidence: 0.35`
4. `startCamera()` — requests webcam, awaits `loadedmetadata`, plays video, starts the rAF loop
5. `stopCamera()` — tears down stream, cancels rAF, clears canvas
6. `tick()` (rAF loop):
   - waits for a fresh video frame
   - avoids overlapping inference with `inferenceBusyRef`
   - calls `await pose.send({ image: video })`
7. `pose.onResults(...)`:
   - draws mirrored video onto the canvas
   - draws skeleton lines + dots over the frame
   - updates `poseData` every other callback

**Key types exported:**
- `Joint` — `{ x, y, z, visibility, screenX, screenY }` — normalized coords + pixel position
- `PoseData` — named joints (nose, leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist, leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle) + `raw` array + `capturedAt` timestamp
- `ModelStatus` — `'loading' | 'ready' | 'error'`
- `CameraStatus` — `'idle' | 'starting' | 'live' | 'error'`

**Landmark indices used (MediaPipe Pose):**
```
nose: 0, leftShoulder: 11, rightShoulder: 12,
leftElbow: 13, rightElbow: 14, leftWrist: 15, rightWrist: 16,
leftHip: 23, rightHip: 24, leftKnee: 25, rightKnee: 26,
leftAnkle: 27, rightAnkle: 28
```

**Skeleton connections drawn:**
Left/right shoulder, shoulder–elbow, elbow–wrist, shoulder–hip, hip–hip, hip–knee, knee–ankle, nose–shoulder (both sides).

**Visibility rule:** `(landmark.visibility ?? 1) >= threshold` — landmarks without a visibility value are treated as visible. Threshold is `0.1` for both lines and dots.

**x-coordinate mirroring:** All draw calls use `(1 - landmark.x) * width` to mirror the skeleton to match the selfie-mode video.

**Important runtime note:** if the UI still shows the old generic model error text, the user is not running the current code. Restart `npm run dev` and hard-refresh the browser.

### `src/App.tsx`

- Calls `usePoseTracking()`
- Left side: canvas element showing video + skeleton
- Right side: data panel with `JointRow` components showing live coords
- Start/Stop camera button in the top bar
- Badge still shows model status; `delegate` is currently always `null` in the new pipeline

### `src/App.css` + `src/index.css`

Dark theme (`#07111f` background). CSS variables in `index.css`:
`--bg`, `--panel`, `--panel-alt`, `--border`, `--text-high`, `--text-soft`, `--accent` (`#5cf9aa`), `--accent-dim`.

---

## Config files

### `vite.config.ts`
```ts
plugins: [react()]
```
There is no longer any special Vite workaround for `@mediapipe/tasks-vision`, because that dependency was removed from `main`.

### `package.json`
```json
"dependencies": {
  "@mediapipe/pose": "^0.5.1675469404"
}
```
`@mediapipe/tasks-vision` and the old postinstall source-map stub were removed.

---

## `temp` branch — what's there

The `temp` branch has a much more complex version. Key things that exist there that could be merged back:

- `src/lib/feedback.ts` — squat and pushup form analysis (knee angle, torso lean, elbow angle, body line), rep counting, spoken voice cues
- `src/lib/stage.ts` — procedural puppet character drawn from landmarks (torso quad, rounded limbs, hinge joints, head circle)
- `src/lib/smoother.ts` — 7-frame causal moving average for real-time smoothing + symmetric centered average for export
- `src/lib/exporter.ts` — JSON export and self-contained HTML player export
- Recording/timeline system — record frames with timestamps, playback, export
- Older, more complex pose/exercise app code that is intentionally not on `main`

---

## Likely next steps

1. **Verify runtime behavior in browser** — compile passes, but the user still needs to confirm the live tracker actually detects poses on their machine.
2. **If model load still fails, capture the exact new error text** — the UI now surfaces the real exception instead of a generic message.
3. **Add explicit debug UI if needed** — frame count, callback count, and landmark count would make browser-side diagnosis faster.
4. **Optionally add world-space landmark output later** — `@mediapipe/pose` exposes `poseWorldLandmarks`, but `main` is currently only using screen-space landmarks.
5. **Keep `main` minimal** — do not merge squat/gym/coach logic back unless explicitly requested.

---

## How to resume work in a new chat

1. Read this file
2. Read `README.md`
3. Run `git log --oneline -10` to see recent commits
4. Run `git status` to check for local changes
5. Read `src/hooks/usePoseTracking.ts` — it contains the entire core logic
6. If working with the old features, `git show temp:src/lib/feedback.ts` etc. to read files from the other branch without checking out

---

## Run commands

```bash
# Install dependencies
npm install

# Start dev server
npm run dev
# Open http://localhost:5173

# Type check
./node_modules/.bin/tsc --noEmit

# Build for production
npm run build

# Lint
npm run lint
```
