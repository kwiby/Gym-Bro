# Project Context

Read this file at the start of every new chat to restore current project state.

---

## Project name

Pose Tracker (repo: `Gym_bro`)

## GitHub

- Remote: `https://github.com/AndrewLu-1/Gym_bro.git`
- Active branch: `main`
- `temp` branch still holds older experimental app history

## What this project is

This repo is now primarily a Python-served local workout coaching app.

The backend:
- opens the webcam from Python
- runs MediaPipe Pose Landmarker locally
- draws the skeleton onto each frame
- streams MJPEG to the browser
- scores a selected exercise
- supports a locked one-rep coaching flow
- supports a timed combo-training mode with per-duration highscores
- optionally generates spoken coaching with ElevenLabs

The React/Vite code in `src/` is older and still present, but it is not the main user flow now.

---

## Current state of `main`

### What works

- Python server at `http://127.0.0.1:8000/`
- Browser UI for live camera stream, status, joints, one-rep coaching, and combo training
- Manual exercise selector instead of automatic exercise detection
- Supported exercises:
  - squat
  - pushup
  - bicep curl
  - overhead press
  - situp
  - lunge
- One-rep coaching flow:
  - user starts a rep check
  - app instructs the user to do one rep
  - app captures that rep only
  - feedback locks on screen until the next check starts
- Voice coaching:
  - optional ElevenLabs audio
  - varied praise for good reps
  - varied playful insults for bad reps
  - browser retry path for autoplay-blocked audio
- Combo training mode:
  - session lengths of 1 minute and 2 minutes
  - start-session button
  - 5-second combo window for chaining good reps
  - browser-generated success sound for good combo reps
  - on-screen combo popup effect
  - separate highscores per session length
- Camera startup hardening:
  - tries camera indices `0`, `1`, `2`
  - prefers `cv2.CAP_DSHOW` on Windows
  - waits for real frames before declaring startup success
  - surfaces real startup errors instead of hanging forever on `Starting camera...`

### Current rough edges

- Voice still depends on valid ElevenLabs credentials and browser playback permissions
- Combo highscores are runtime-only; they are not persisted to disk yet
- Exercise scoring is heuristic and angle-based, so edge cases depend on camera angle and visibility
- The older Vite app docs and code still exist in the repo, but they are not the active product surface

---

## Main architecture

### Primary entry point

`web_pose_server.py`

This file now contains almost the entire active app.

### Key responsibilities inside `web_pose_server.py`

1. Serve a browser UI from an inline `HTML_PAGE` string
2. Open and warm up the webcam from Python
3. Run MediaPipe Pose on each frame
4. Draw the pose skeleton and stream the resulting frames to `/stream.mjpg`
5. Expose current state as JSON at `/pose.json`
6. Accept browser POST actions for:
   - `/exercise`
   - `/rep-check/start`
   - `/session/duration`
   - `/session/start`
7. Manage spoken feedback and audio bytes for `/coach-audio.mp3`

### Important classes

#### `WorkoutTracker`

Handles the one-rep coaching flow.

- Tracks selected exercise
- Tracks rep count
- Holds coach state: `idle`, `waiting`, `collecting`, `complete`
- Collects analysis samples only for the current rep check
- Locks and returns the final rep feedback after a completed rep

#### `SessionTracker`

Handles timed combo training.

- Tracks selected session duration
- Tracks active/inactive session state
- Tracks combo count, current session best, and per-duration highscores
- Requires a good rep within `COMBO_WINDOW_MS` to continue the combo
- Emits combo event versions so the browser can trigger the popup/sound once per event

#### `VoiceManager`

Handles optional ElevenLabs speech generation.

- Stores latest generated MP3 bytes
- Exposes a version number so the browser knows when fresh audio exists
- Avoids duplicate generation for identical text

#### `PoseRuntime`

Stores the latest frame and JSON payload shared across the HTTP handlers.

---

## Active UI behavior

The current UI is browser-rendered from the inline HTML inside `web_pose_server.py`.

It includes:

- live MJPEG stream image
- status pill
- frame size / timestamp / landmark count
- selected exercise and rep count
- exercise dropdown with black text on a light background
- voice toggle
- combo session duration dropdown with black text
- combo session start button
- live combo, session best, time remaining, and highscores
- coach instruction box for the one-rep flow
- start-one-rep button
- analysis card that stays locked after the rep completes
- joint coordinate list
- animated combo burst overlay
- browser-generated combo success sound

---

## Exercise analysis model

The system no longer tries to infer which exercise the user is doing.

Instead:
- the user chooses the exercise manually
- the backend runs the analyzer for that exercise only

Current analyzers in `web_pose_server.py`:

- `analyze_squat`
- `analyze_pushup`
- `analyze_bicep_curl`
- `analyze_overhead_press`
- `analyze_situp`
- `analyze_lunge`

Each analyzer returns:
- `exercise`
- `exerciseLabel`
- `stage`
- `status`
- `metrics`
- `feedback` with `tone`, `title`, and `details`

`tone` drives voice style and combo scoring:
- `good` can continue combo chains
- `warn` and `bad` trigger playful insult voice feedback in the one-rep flow
- `neutral` is informational

---

## Camera startup behavior

The app uses `open_camera()` in `web_pose_server.py`.

Current logic:
- try indices `0`, `1`, `2`
- on Windows, try `cv2.CAP_DSHOW` first
- wait up to `CAMERA_WARMUP_FRAMES` reads for real frames
- if all attempts fail, surface a concrete error string in the UI payload

If the page still says `Searching for pose...`, that usually means the camera is running but no valid body landmarks are being found yet.

---

## Files worth reading first

1. `web_pose_server.py`
2. `README.md`
3. `git log --oneline -10`
4. `git status`

Only read `src/` if the task is specifically about the older Vite tracker.

---

## Run commands

### Run the active Python app

```cmd
cd C:\Users\andre\Gym_bro
.venv\Scripts\python.exe web_pose_server.py
```

Optional voice:

```cmd
set ELEVENLABS_API_KEY=your_key_here
.venv\Scripts\python.exe web_pose_server.py
```

Open:

```text
http://127.0.0.1:8000/
```

### Syntax check

```cmd
.venv\Scripts\python.exe -m py_compile web_pose_server.py
```

### Older Vite app

```bash
npm install
npm run dev
```

---

## Resume checklist for a new chat

1. Read this file
2. Read `README.md`
3. Run `git status`
4. Run `git log --oneline -10`
5. Read `web_pose_server.py`
6. Ignore the older `src/` app unless the user explicitly asks about it
