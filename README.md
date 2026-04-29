# Pose Tracker

A local pose-tracking and exercise-coaching app built with MediaPipe. The current primary flow is a Python web server that runs pose detection locally, streams the annotated camera feed into the browser, and gives angle-based form feedback every 5 seconds.

## What it does

- Opens your webcam from Python and runs MediaPipe Pose Landmarker locally
- Streams the processed camera feed to the browser with a live skeleton overlay
- Tracks key joints and displays their live coordinates in the UI
- Detects common exercises from pose shape: squat, pushup, and bicep curl
- Scores form from landmark angles and publishes a summary every 5 seconds from the previous 5-second window
- Optionally generates spoken correction cues with ElevenLabs

## Current app modes

### Python local web app

This is the main workflow right now.

- Entry point: `web_pose_server.py`
- Browser URL: `http://127.0.0.1:8000/`
- Uses `pose_landmarker_lite.task` directly from Python
- Supports optional ElevenLabs voice coaching

### Vite browser app

The repo also still contains the older React/Vite browser pose tracker in `src/`, but the current exercise-coaching flow is the Python server above.

## Running locally

### Python app

Create or activate the local venv, then run:

```cmd
cd C:\Users\andre\Gym_bro
set ELEVENLABS_API_KEY=your_key_here
.venv\Scripts\python.exe web_pose_server.py
```

Then open `http://127.0.0.1:8000/`.

If you do not want voice coaching, omit `ELEVENLABS_API_KEY`.

### Optional ElevenLabs settings

```cmd
set ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
set ELEVENLABS_MODEL_ID=eleven_multilingual_v2
```

### Vite app

```bash
npm install
npm run dev
```

Then open `http://localhost:5173/`.

## How feedback works

- Pose samples are gathered continuously while the camera is live
- The UI updates the live video and joint panel continuously
- Form analysis is summarized every 5 seconds using the previous 5 seconds of samples
- The summary includes:
  - detected exercise
  - averaged angle metrics
  - most common form issues in that window
  - optional spoken coaching audio

## Tech stack

- **Python + OpenCV** for webcam capture and MJPEG streaming
- **MediaPipe Tasks** for pose landmark detection
- **Standard library HTTP server** for the local web app
- **ElevenLabs Python SDK** for optional spoken feedback
- **React 19 + TypeScript + Vite** for the older browser-only app still in the repo

## Project structure

```text
web_pose_server.py         Main local pose server with exercise coaching and voice
pose_landmarker_lite.task  MediaPipe pose model used by the Python flow
mediapipe_handler.py       Earlier standalone Python camera script
src/                       Older React/Vite browser pose tracker
README.md                  Project overview and run instructions
PROJECT_CONTEXT.md         Session handoff and architecture notes
```

## Form-coaching exercises

- Squat: knee angle, hip angle, torso lean
- Pushup: elbow angle, body line angle
- Bicep curl: elbow angle, upper-arm drift

## Tips for better tracking

- Stand far enough back to keep your full body visible
- Use good lighting
- Turn slightly side-on for cleaner joint-angle scoring
- Avoid running another app that already owns the webcam

## Branches

| Branch | Contents |
|--------|----------|
| `main` | Current local pose-tracking and coaching app |
| `temp` | Earlier version with more experimental coaching, MoCap, export, and recording features |
