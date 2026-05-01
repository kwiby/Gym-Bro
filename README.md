<p align="center">
  <img src="https://raw.githubusercontent.com/AndrewLu-1/Gym_bro/main/Gym%20Bro%20Icon.png" width="300" />
</p>

# Gym Bro - A [ConHacks 2026](https://conhacks.io/) Hackathon Project

**Devpost:** https://devpost.com/software/gym-bro-m23hli?ref_content=my-projects-tab&ref_feature=my_projects

While working out, have you ever felt that your form had less structural integrity than a piece of wet toast? Well, worry no more for we have a solution, Gym Bro!
![Screenshot 1](https://github.com/AndrewLu-1/Gym_bro/blob/main/Screenshot%201.png)

## What It Does
Gym Bro is a workout coaching website that uses your webcam to analyze your workout form in real time.

## How It Was Built

<table style="border: none; border-collapse: collapse;">
  <tr>
    <!-- LEFT COLUMN -->
    <td style="border: none; vertical-align: top;">

**Tech Stack:**
- Python.
- OpenCV for webcam capture.
- MediaPipe pose landmarker for pose tracking.
- Python http.server for the local web server.
- HTML/CSS/JavaScript for the browser UI.
- **ElevenLabs** text-to-speech for live voice coaching.

**Architecture:**
- Python manages the webcam and pose processing.
- Frames are analyzed locally.
- The annotated feed is streamed into the browser.
- A JSON endpoint exposes exercise state, rep counts, combo state, and feedback.
- The browser renders the live UI and handles optional voice playback.

    </td>

    <!-- RIGHT COLUMN -->
    <td style="border: none; vertical-align: top;">
      <img src="https://github.com/AndrewLu-1/Gym_bro/blob/main/Screenshot%203.png" width="250">
    </td>
  </tr>
</table>

## How To Use It

<table style="border: none; border-collapse: collapse;">
  <tr>
    <!-- LEFT COLUMN -->
    <td style="border: none; vertical-align: top;">

### One-Rep Check
1. Select an exercise from the dropdown.
2. Click `Enable voice` if you want spoken coaching.
3. Click `Start one-rep check`.
4. Perform exactly one full rep.
5. The app locks the feedback for that rep on screen until you start another check.

### Combo Session
1. Select an exercise.
2. Choose a session length: `1 minute` or `2 minutes`.
3. Click `Start combo session`.
4. String together good reps.
5. A good rep adds to the combo if the next good rep lands within 5 seconds.
6. The UI shows live combo, session best, time remaining, and separate high scores for each session length.

    </td>

    <!-- RIGHT COLUMN -->
    <td style="border: none; vertical-align: top;">
      <img src="https://github.com/AndrewLu-1/Gym_bro/blob/main/Screenshot%202.png" width="250">
    </td>
  </tr>
</table>

## Tech Stack

- **Python + OpenCV** for webcam capture and MJPEG streaming.
- **MediaPipe Tasks** for pose landmark detection
- **Standard library HTTP server** for the local web app
- **ElevenLabs Python SDK** for optional spoken coaching
- **Vanilla browser JS/CSS inside the served HTML** for the UI

## Project Structure

```text
web_pose_server.py         Main local pose server with exercise coaching.
pose_landmarker_lite.task  MediaPipe pose model used by the Python flow.
mediapipe_handler.py       Skeleton standalone Python camera script.
```
