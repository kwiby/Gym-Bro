# Pose Tracker

A local pose-tracking and exercise-coaching app built around a Python web server. The backend owns the webcam, runs MediaPipe Pose locally, streams the annotated feed into the browser, and layers on two coaching modes: locked one-rep form checks and timed combo sessions.

## What it does

- Opens your webcam from Python and runs MediaPipe Pose Landmarker locally
- Streams the processed camera feed to the browser with a live skeleton overlay
- Lets you manually choose the exercise to score from a dropdown
- Supports: squat, pushup, bicep curl, overhead press, situp, and lunge
- Runs a one-rep coaching flow that locks feedback after a single completed rep
- Runs timed combo sessions with a 5-second combo window, session best, and per-duration high scores
- Plays optional ElevenLabs spoken coaching with varied praise and playful insults
- Plays a browser-generated success sound and combo popup for good reps during combo sessions

## Main workflow

The primary app is the Python server in `web_pose_server.py`.

- Entry point: `web_pose_server.py`
- Browser URL: `http://127.0.0.1:8000/`
- MediaPipe model: `pose_landmarker_lite.task`
- Optional voice: ElevenLabs via `ELEVENLABS_API_KEY`

The older React/Vite browser tracker still exists in `src/`, but it is not the main coaching flow now.

## Running locally

### Python app

From the repo root:

```cmd
cd C:\Users\andre\Gym_bro
.venv\Scripts\python.exe web_pose_server.py
```

Then open:

```text
http://127.0.0.1:8000/
```

### With voice coaching

```cmd
cd C:\Users\andre\Gym_bro
set ELEVENLABS_API_KEY=your_key_here
.venv\Scripts\python.exe web_pose_server.py
```

Optional voice configuration:

```cmd
set ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
set ELEVENLABS_MODEL_ID=eleven_multilingual_v2
```

### Syntax check

```cmd
.venv\Scripts\python.exe -m py_compile web_pose_server.py
```

## How to use it

### One-rep coaching

1. Select an exercise from the dropdown.
2. Click `Enable voice` if you want spoken coaching.
3. Click `Start one-rep check`.
4. Perform exactly one full rep.
5. The app locks the feedback for that rep on screen until you start another check.

### Combo training

1. Select an exercise.
2. Choose a session length: `1 minute` or `2 minutes`.
3. Click `Start combo session`.
4. String together good reps.
5. A good rep adds to the combo if the next good rep lands within 5 seconds.
6. The UI shows live combo, session best, time remaining, and separate high scores for each session length.

## Camera behavior

The Python server now tries multiple camera openings on Windows to avoid silent startup hangs.

- tries camera indices `0`, `1`, and `2`
- prefers `cv2.CAP_DSHOW` on Windows
- waits for real frames before declaring success
- surfaces a real camera error instead of hanging forever on `Starting camera...`

If the page still cannot start the camera:

1. Close Zoom, Discord, Teams, OBS, and the Windows Camera app.
2. Make sure Windows camera permissions allow desktop apps.
3. Restart the Python server.

## Tech stack

- **Python + OpenCV** for webcam capture and MJPEG streaming
- **MediaPipe Tasks** for pose landmark detection
- **Standard library HTTP server** for the local web app
- **ElevenLabs Python SDK** for optional spoken coaching
- **Vanilla browser JS/CSS inside the served HTML** for the current UI
- **React 19 + TypeScript + Vite** for the older browser-only tracker still in the repo

## Project structure

```text
web_pose_server.py         Main local pose server with exercise coaching
pose_landmarker_lite.task  MediaPipe pose model used by the Python flow
mediapipe_handler.py       Earlier standalone Python camera script
src/                       Older React/Vite browser pose tracker
README.md                  Project overview and run instructions
PROJECT_CONTEXT.md         Session handoff and architecture notes
```

## Scored exercises

- Squat: knee angle, hip angle, torso lean
- Pushup: elbow angle, body line
- Bicep curl: elbow angle, upper-arm drift
- Overhead press: elbow angle, arm stack
- Situp: torso angle, hip angle
- Lunge: knee angle, torso lean, hip angle

## Tracking tips

- Stand far enough back to keep your whole body visible
- Use decent lighting
- Turn slightly side-on when possible so joint angles score more clearly
- Avoid other apps owning the webcam

## Branches

| Branch | Contents |
|--------|----------|
| `main` | Current Python pose-tracking and coaching app |
| `temp` | Earlier experimental coaching, MoCap, export, and recording work |
