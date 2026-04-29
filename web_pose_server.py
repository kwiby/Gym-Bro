import json
import math
import os
import random
import threading
import time
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import mediapipe as mp
from elevenlabs.client import ElevenLabs


MODEL_PATH = 'pose_landmarker_lite.task'
HOST = '127.0.0.1'
PORT = 8000
FRAME_BOUNDARY = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
CAMERA_INDICES = (0, 1, 2)
CAMERA_WARMUP_FRAMES = 30
COMBO_WINDOW_MS = 5000
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', '21m00Tcm4TlvDq8ikWAM')
ELEVENLABS_MODEL_ID = os.getenv('ELEVENLABS_MODEL_ID', 'eleven_multilingual_v2')
SUPPORTED_EXERCISES = {
    'squat': 'Squat',
    'pushup': 'Pushup',
    'bicep_curl': 'Bicep curl',
    'overhead_press': 'Overhead press',
    'situp': 'Situp',
    'lunge': 'Lunge',
}
DEFAULT_EXERCISE = 'squat'
SESSION_DURATION_OPTIONS = {
    '60': {'label': '1 minute', 'seconds': 60},
    '120': {'label': '2 minutes', 'seconds': 120},
}
SUPPORTED_EXERCISE_OPTIONS = [
    {'value': value, 'label': label}
    for value, label in SUPPORTED_EXERCISES.items()
]
SESSION_DURATION_SELECT_OPTIONS = [
    {'value': value, 'label': option['label']}
    for value, option in SESSION_DURATION_OPTIONS.items()
]
GOOD_FORM_LINES = [
    'That rep looked sharp. Keep stacking clean ones like that.',
    'Very solid form. That is the kind of rep worth repeating.',
    'Nice work. You moved that rep with real control.',
    'That was clean. Keep owning the movement like that.',
    'Strong rep. You actually made that look athletic.',
    'Good form there. Stay locked in and repeat it.',
    'That was crisp. Keep the same rhythm on the next one.',
    'Excellent rep. Controlled, clean, and confident.',
    'That looked legit. More of that, less chaos.',
]
BAD_FORM_LINES = [
    'That rep was a little suspicious. Let us aim for exercise, not interpretive dance.',
    'Bold attempt. Your form looked like it was assembled during a power outage.',
    'That was rough. I have seen sturdier mechanics in a shopping cart.',
    'Not your cleanest work. That rep had the structural integrity of wet toast.',
    'That form wandered off halfway through the movement. Please bring it back.',
    'Messy rep. Your joints were freelancing out there.',
    'That one looked confused. Commit to the movement before it files a complaint.',
    'Yikes. That rep had all the grace of a folding chair in a windstorm.',
    'That form needs help. Right now it is more chaos than coaching success.',
]
POLL_NAMES = {
    0: 'nose',
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
}

VISIBILITY_THRESHOLD = 0.35


def exercise_label(exercise: str) -> str:
    return SUPPORTED_EXERCISES.get(exercise, 'Exercise')


def build_rep_prompt(exercise: str) -> str:
    label = exercise_label(exercise)
    return f"Do exactly one {label.lower()} rep. I will watch that one rep, then lock your feedback so it stays clear."


def session_duration_label(duration_key: str) -> str:
    return SESSION_DURATION_OPTIONS.get(duration_key, SESSION_DURATION_OPTIONS['60'])['label']


def open_camera() -> tuple[Any | None, str | None]:
    attempts: list[str] = []

    for index in CAMERA_INDICES:
        api_preferences = [cv2.CAP_DSHOW] if os.name == 'nt' else [0]
        if os.name == 'nt':
            api_preferences.append(0)

        for api_preference in api_preferences:
            cap = cv2.VideoCapture(index, api_preference) if api_preference else cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                attempts.append(f'camera {index} did not open')
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            for _ in range(CAMERA_WARMUP_FRAMES):
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, 'size', 0) > 0:
                    return cap, None
                time.sleep(0.05)

            cap.release()
            attempts.append(f'camera {index} opened but never returned a frame')

    if not attempts:
        return None, 'No camera devices were attempted.'

    return None, 'Could not start webcam. Tried: ' + '; '.join(attempts)

INDEX_TO_NAME = {
    0: 'nose',
    1: 'left_eye_inner',
    2: 'left_eye',
    3: 'left_eye_outer',
    4: 'right_eye_inner',
    5: 'right_eye',
    6: 'right_eye_outer',
    7: 'left_ear',
    8: 'right_ear',
    9: 'mouth_left',
    10: 'mouth_right',
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    17: 'left_pinky',
    18: 'right_pinky',
    19: 'left_index',
    20: 'right_index',
    21: 'left_thumb',
    22: 'right_thumb',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
    29: 'left_heel',
    30: 'right_heel',
    31: 'left_foot_index',
    32: 'right_foot_index',
}

HTML_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Python Pose Tracker</title>
    <style>
      :root {
        color-scheme: dark;
        font-family: Inter, system-ui, sans-serif;
        background: #07111f;
        color: #f4f7fb;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        background: radial-gradient(circle at top, #10233b, #07111f 55%);
      }
      .shell {
        width: min(1180px, calc(100vw - 32px));
        margin: 24px auto;
        display: grid;
        gap: 20px;
        grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr);
      }
      .panel {
        background: rgba(7, 17, 31, 0.82);
        border: 1px solid rgba(92, 249, 170, 0.14);
        border-radius: 24px;
        overflow: hidden;
        box-shadow: 0 22px 48px rgba(0, 0, 0, 0.32);
      }
      .hero {
        padding: 20px 22px 0;
      }
      .eyebrow {
        color: #5cf9aa;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.18em;
      }
      h1 {
        margin: 10px 0 8px;
        font-size: clamp(28px, 4vw, 44px);
        line-height: 1.05;
      }
      p {
        margin: 0 0 18px;
        color: #a9b7c8;
        line-height: 1.5;
      }
      .stream-wrap {
        position: relative;
        margin: 0 22px 22px;
        background: #02070d;
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
      }
      .combo-burst {
        position: absolute;
        left: 50%;
        top: 18%;
        transform: translate(-50%, -50%) scale(0.8);
        padding: 14px 20px;
        border-radius: 999px;
        background: rgba(92, 249, 170, 0.92);
        color: #07111f;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        opacity: 0;
        pointer-events: none;
        box-shadow: 0 18px 40px rgba(92, 249, 170, 0.35);
      }
      .combo-burst.show {
        animation: combo-pop 900ms ease forwards;
      }
      @keyframes combo-pop {
        0% { opacity: 0; transform: translate(-50%, -50%) scale(0.72); }
        18% { opacity: 1; transform: translate(-50%, -50%) scale(1.08); }
        70% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        100% { opacity: 0; transform: translate(-50%, -64%) scale(1.04); }
      }
      img {
        width: 100%;
        display: block;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        background: #02070d;
      }
      .sidebar {
        padding: 20px;
      }
      .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(92, 249, 170, 0.08);
        color: #bdfcd9;
        font-size: 14px;
      }
      .status.error {
        background: rgba(255, 120, 120, 0.08);
        color: #ffb3b3;
      }
      .dot {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: currentColor;
      }
      .stats {
        margin: 18px 0;
        display: grid;
        gap: 10px;
      }
      .stat {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.04);
        color: #dce6f2;
      }
      .joint-list {
        display: grid;
        gap: 8px;
        max-height: 36vh;
        overflow: auto;
      }
      .analysis {
        margin-bottom: 18px;
        padding: 14px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.04);
      }
      .coach-box {
        margin: 0 0 16px;
        padding: 14px;
        border-radius: 16px;
        border: 1px solid rgba(143, 184, 255, 0.28);
        background: rgba(143, 184, 255, 0.08);
      }
      .coach-box h2 {
        margin: 0 0 6px;
        font-size: 18px;
      }
      .analysis.good {
        border: 1px solid rgba(92, 249, 170, 0.28);
      }
      .analysis.warn {
        border: 1px solid rgba(255, 196, 107, 0.28);
      }
      .analysis.bad {
        border: 1px solid rgba(255, 120, 120, 0.28);
      }
      .analysis.neutral {
        border: 1px solid rgba(255, 255, 255, 0.1);
      }
      .analysis h2 {
        margin: 0 0 6px;
        font-size: 20px;
      }
      .voice-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0 0 16px;
        flex-wrap: wrap;
      }
      .control-block {
        margin: 0 0 16px;
      }
      .control-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin: 0 0 16px;
      }
      .control-label {
        display: block;
        margin: 0 0 8px;
        color: #a9b7c8;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .exercise-select {
        width: 100%;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 12px 14px;
        font: inherit;
        color: #07111f;
        background: #f4f7fb;
      }
      .session-box {
        margin: 0 0 16px;
        padding: 14px;
        border-radius: 16px;
        border: 1px solid rgba(255, 196, 107, 0.24);
        background: rgba(255, 196, 107, 0.08);
      }
      .session-box h2 {
        margin: 0 0 6px;
        font-size: 18px;
      }
      .session-stats {
        display: grid;
        gap: 8px;
        margin: 12px 0 0;
      }
      .session-stat {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        color: #dce6f2;
        font-size: 13px;
      }
      .high-score-list {
        margin: 12px 0 0;
        padding-left: 18px;
        color: #dce6f2;
      }
      .voice-button {
        border: 0;
        border-radius: 999px;
        padding: 10px 14px;
        font: inherit;
        color: #07111f;
        background: #5cf9aa;
        cursor: pointer;
      }
      .voice-button[disabled] {
        opacity: 0.5;
        cursor: not-allowed;
      }
      .coach-button {
        width: 100%;
        margin: 0 0 16px;
        border: 0;
        border-radius: 14px;
        padding: 12px 14px;
        font: inherit;
        font-weight: 600;
        color: #07111f;
        background: linear-gradient(135deg, #5cf9aa, #8fb8ff);
        cursor: pointer;
      }
      .coach-button[disabled] {
        opacity: 0.55;
        cursor: not-allowed;
      }
      .session-button {
        width: 100%;
        margin: 0 0 16px;
        border: 0;
        border-radius: 14px;
        padding: 12px 14px;
        font: inherit;
        font-weight: 700;
        color: #07111f;
        background: linear-gradient(135deg, #ffc46b, #ff8f8f);
        cursor: pointer;
      }
      .session-button[disabled] {
        opacity: 0.55;
        cursor: not-allowed;
      }
      .analysis p {
        margin: 0;
      }
      .feedback-list,
      .metric-list {
        margin: 12px 0 0;
        padding-left: 18px;
        color: #dce6f2;
      }
      .metric-list {
        color: #a9b7c8;
      }
      .joint {
        padding: 10px 12px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.04);
        font-size: 13px;
        line-height: 1.4;
      }
      .muted { color: #8ea0b5; }
      @media (max-width: 900px) {
        .shell { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <section class="panel">
        <div class="hero">
          <div class="eyebrow">Python MediaPipe</div>
          <h1>Working local pose tracker</h1>
          <p>
            This page is served by Python. The backend owns the webcam, runs MediaPipe
            pose detection, draws the skeleton, and streams the result into the browser.
          </p>
        </div>
        <div class="stream-wrap">
          <div id="combo-burst" class="combo-burst"></div>
          <img src="/stream.mjpg" alt="Live pose stream" />
        </div>
      </section>

      <aside class="panel sidebar">
        <div id="status" class="status"><span class="dot"></span><span>Starting camera...</span></div>
        <div class="stats">
          <div class="stat"><span>Exercise</span><strong id="selected-exercise">Squat</strong></div>
          <div class="stat"><span>Rep count</span><strong id="rep-count">0</strong></div>
          <div class="stat"><span>Frame size</span><strong id="frame-size">-</strong></div>
          <div class="stat"><span>Last update</span><strong id="timestamp">-</strong></div>
          <div class="stat"><span>Pose landmarks</span><strong id="landmark-count">0</strong></div>
        </div>
        <div class="control-block">
          <label class="control-label" for="exercise-select">Exercise to score</label>
          <select id="exercise-select" class="exercise-select"></select>
        </div>
        <div class="control-grid">
          <div class="control-block">
            <label class="control-label" for="session-duration-select">Combo session length</label>
            <select id="session-duration-select" class="exercise-select"></select>
          </div>
          <div class="control-block">
            <label class="control-label">High score</label>
            <div class="stat"><span>Best combo</span><strong id="combo-high-score">0</strong></div>
          </div>
        </div>
        <div class="voice-row">
          <button id="voice-toggle" class="voice-button" type="button">Enable voice</button>
          <span id="voice-status" class="muted">Voice unavailable</span>
        </div>
        <section id="session-box" class="session-box">
          <h2>Combo Training</h2>
          <p id="session-message">Start a timed session and chain good reps within 5 seconds to build your combo.</p>
          <div class="session-stats">
            <div class="session-stat"><span>Live combo</span><strong id="combo-count">0</strong></div>
            <div class="session-stat"><span>Session best</span><strong id="combo-best">0</strong></div>
            <div class="session-stat"><span>Time left</span><strong id="session-time-remaining">--:--</strong></div>
          </div>
          <ul id="high-score-list" class="high-score-list"></ul>
        </section>
        <button id="session-button" class="session-button" type="button">Start combo session</button>
        <section id="coach-box" class="coach-box">
          <h2 id="coach-title">Pick an exercise, then start a one-rep check.</h2>
          <p id="coach-message">Do exactly one rep when the app is watching. It will lock the feedback after that rep so nothing gets overwritten.</p>
        </section>
        <button id="coach-button" class="coach-button" type="button">Start one-rep check</button>
        <section id="analysis" class="analysis neutral">
          <div class="muted">Your locked rep feedback will appear here after the rep finishes.</div>
        </section>
        <div id="joint-list" class="joint-list"></div>
      </aside>
    </div>

    <script>
      const statusEl = document.getElementById('status')
      const frameSizeEl = document.getElementById('frame-size')
      const timestampEl = document.getElementById('timestamp')
      const landmarkCountEl = document.getElementById('landmark-count')
      const selectedExerciseEl = document.getElementById('selected-exercise')
      const repCountEl = document.getElementById('rep-count')
      const comboBurstEl = document.getElementById('combo-burst')
      const exerciseSelectEl = document.getElementById('exercise-select')
      const sessionDurationSelectEl = document.getElementById('session-duration-select')
      const comboHighScoreEl = document.getElementById('combo-high-score')
      const comboCountEl = document.getElementById('combo-count')
      const comboBestEl = document.getElementById('combo-best')
      const sessionTimeRemainingEl = document.getElementById('session-time-remaining')
      const sessionMessageEl = document.getElementById('session-message')
      const highScoreListEl = document.getElementById('high-score-list')
      const sessionButtonEl = document.getElementById('session-button')
      const coachTitleEl = document.getElementById('coach-title')
      const coachMessageEl = document.getElementById('coach-message')
      const coachButtonEl = document.getElementById('coach-button')
      const analysisEl = document.getElementById('analysis')
      const jointListEl = document.getElementById('joint-list')
      const voiceToggleEl = document.getElementById('voice-toggle')
      const voiceStatusEl = document.getElementById('voice-status')
      const audioEl = new Audio()
      let voiceEnabled = false
      let lastVoiceVersion = 0
      let latestVoiceVersion = 0
      let optionSignature = ''
      let durationOptionSignature = ''
      let queuedVoiceVersion = 0
      let voiceStatusOverride = ''
      let lastComboEventVersion = 0
      let comboBurstTimeout = 0
      let audioContext = null

      function updateVoiceStatus(serverStatus) {
        voiceStatusEl.textContent = voiceStatusOverride || serverStatus || 'Voice unavailable'
      }

      voiceToggleEl.addEventListener('click', async () => {
        voiceEnabled = !voiceEnabled
        voiceToggleEl.textContent = voiceEnabled ? 'Disable voice' : 'Enable voice'
        if (voiceEnabled) {
          voiceStatusOverride = ''
          try {
            audioEl.muted = false
            if (latestVoiceVersion && latestVoiceVersion !== lastVoiceVersion) {
              queuedVoiceVersion = 0
              await playVoiceVersion(latestVoiceVersion)
            }
          } catch (error) {
          }
        } else {
          audioEl.pause()
        }
      })

      function renderJoint(name, joint) {
        return `<div class="joint"><strong>${name}</strong><br><span class="muted">x ${joint.x.toFixed(3)} · y ${joint.y.toFixed(3)} · z ${joint.z.toFixed(3)} · vis ${Math.round(joint.visibility * 100)}%</span></div>`
      }

      function formatRemaining(ms) {
        const totalSeconds = Math.max(0, Math.ceil((ms || 0) / 1000))
        const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0')
        const seconds = String(totalSeconds % 60).padStart(2, '0')
        return `${minutes}:${seconds}`
      }

      function syncDurationOptions(payload) {
        const options = payload.sessionDurationOptions || []
        const nextSignature = JSON.stringify(options)
        if (nextSignature !== durationOptionSignature) {
          durationOptionSignature = nextSignature
          sessionDurationSelectEl.innerHTML = options
            .map((option) => `<option value="${option.value}">${option.label}</option>`)
            .join('')
        }

        if (payload.sessionDurationKey) {
          sessionDurationSelectEl.value = payload.sessionDurationKey
        }
      }

      function playComboSound() {
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext
        if (!AudioContextCtor) {
          return
        }

        if (!audioContext) {
          audioContext = new AudioContextCtor()
        }

        if (audioContext.state === 'suspended') {
          audioContext.resume().catch(() => {})
        }

        const now = audioContext.currentTime
        const oscillator = audioContext.createOscillator()
        const gain = audioContext.createGain()
        oscillator.type = 'triangle'
        oscillator.frequency.setValueAtTime(660, now)
        oscillator.frequency.exponentialRampToValueAtTime(990, now + 0.14)
        gain.gain.setValueAtTime(0.0001, now)
        gain.gain.exponentialRampToValueAtTime(0.18, now + 0.02)
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.24)
        oscillator.connect(gain)
        gain.connect(audioContext.destination)
        oscillator.start(now)
        oscillator.stop(now + 0.25)
      }

      function showComboBurst(comboCount, exerciseLabel) {
        window.clearTimeout(comboBurstTimeout)
        comboBurstEl.classList.remove('show')
        void comboBurstEl.offsetWidth
        comboBurstEl.textContent = comboCount > 1 ? `${comboCount}x combo` : `${exerciseLabel} clean hit`
        comboBurstEl.classList.add('show')
        comboBurstTimeout = window.setTimeout(() => {
          comboBurstEl.classList.remove('show')
        }, 950)
        playComboSound()
      }

      function renderHighScores(highScores, selectedDurationKey) {
        const options = [
          { key: '60', label: '1 minute' },
          { key: '120', label: '2 minutes' },
        ]
        highScoreListEl.innerHTML = options
          .map((option) => {
            const score = highScores?.[option.key] || 0
            const active = option.key === selectedDurationKey ? ' <span class="muted">(selected)</span>' : ''
            return `<li>${option.label}: <strong>${score}</strong>${active}</li>`
          })
          .join('')
      }

      async function playVoiceVersion(version) {
        if (!version) {
          return
        }

        audioEl.src = `/coach-audio.mp3?v=${version}`
        try {
          await audioEl.play()
          lastVoiceVersion = version
          voiceStatusOverride = ''
        } catch (error) {
          voiceStatusOverride = 'Voice ready - browser blocked autoplay, click Enable voice again.'
          updateVoiceStatus('')
        }
      }

      audioEl.addEventListener('ended', async () => {
        if (!voiceEnabled || !queuedVoiceVersion || queuedVoiceVersion === lastVoiceVersion) {
          return
        }

        const nextVersion = queuedVoiceVersion
        queuedVoiceVersion = 0
        await playVoiceVersion(nextVersion)
      })

      function syncExerciseOptions(payload) {
        const options = payload.supportedExercises || []
        const nextSignature = JSON.stringify(options)
        if (nextSignature !== optionSignature) {
          optionSignature = nextSignature
          exerciseSelectEl.innerHTML = options
            .map((option) => `<option value="${option.value}">${option.label}</option>`)
            .join('')
        }

        if (payload.selectedExercise) {
          exerciseSelectEl.value = payload.selectedExercise
        }
      }

      function renderAnalysis(analysis) {
        if (!analysis) {
          return '<div class="muted">No locked rep feedback yet.</div>'
        }

        const windowLabel = analysis.windowStartedAt && analysis.windowEndedAt
          ? `Window: ${new Date(analysis.windowStartedAt).toLocaleTimeString()} - ${new Date(analysis.windowEndedAt).toLocaleTimeString()}`
          : ''
        const details = (analysis.feedback?.details || [])
          .map((detail) => `<li>${detail}</li>`)
          .join('')
        const metrics = (analysis.metrics || [])
          .map((metric) => `<li>${metric.label}: ${metric.value.toFixed(1)} deg <span class="muted">(${metric.target})</span></li>`)
          .join('')

        return `
          <div class="eyebrow">${analysis.exerciseLabel || 'Exercise'}</div>
          <h2>${analysis.feedback?.title || analysis.status || 'Analysis'}</h2>
          <p>${analysis.status || ''}</p>
          ${windowLabel ? `<p class="muted">${windowLabel}</p>` : ''}
          ${metrics ? `<ul class="metric-list">${metrics}</ul>` : ''}
          ${details ? `<ul class="feedback-list">${details}</ul>` : ''}
        `
      }

      async function poll() {
        try {
          const response = await fetch('/pose.json', { cache: 'no-store' })
          const payload = await response.json()

          statusEl.className = payload.error ? 'status error' : 'status'
          statusEl.innerHTML = `<span class="dot"></span><span>${payload.error || payload.status}</span>`
          frameSizeEl.textContent = payload.frameWidth && payload.frameHeight ? `${payload.frameWidth} x ${payload.frameHeight}` : '-'
          timestampEl.textContent = payload.updatedAt ? new Date(payload.updatedAt).toLocaleTimeString() : '-'
          landmarkCountEl.textContent = String(payload.landmarkCount || 0)
          selectedExerciseEl.textContent = payload.selectedExerciseLabel || 'Exercise'
          repCountEl.textContent = String(payload.repCount || 0)
          comboCountEl.textContent = String(payload.comboCount || 0)
          comboBestEl.textContent = String(payload.comboBestCurrent || 0)
          comboHighScoreEl.textContent = String((payload.comboHighScores || {})[payload.sessionDurationKey] || 0)
          sessionTimeRemainingEl.textContent = formatRemaining(payload.sessionRemainingMs)
          sessionMessageEl.textContent = payload.sessionActive
            ? `Session live. Hit another good rep within 5 seconds to keep the combo climbing.`
            : `Start a ${payload.sessionDurationLabel || '1 minute'} session and chain good reps within 5 seconds to build your combo.`
          sessionButtonEl.textContent = payload.sessionActive
            ? `Session running: ${formatRemaining(payload.sessionRemainingMs)}`
            : 'Start combo session'
          sessionButtonEl.disabled = Boolean(payload.sessionActive)
          coachTitleEl.textContent = payload.coachTitle || 'One-rep coaching'
          coachMessageEl.textContent = payload.coachMessage || ''
          coachButtonEl.textContent = payload.coachButtonLabel || 'Start one-rep check'
          coachButtonEl.disabled = Boolean(payload.coachButtonDisabled)
          syncExerciseOptions(payload)
          syncDurationOptions(payload)
          renderHighScores(payload.comboHighScores, payload.sessionDurationKey)
          analysisEl.className = `analysis ${payload.analysis?.feedback?.tone || 'neutral'}`
          analysisEl.innerHTML = renderAnalysis(payload.analysis)
          voiceToggleEl.disabled = !payload.voice?.available
          latestVoiceVersion = payload.voice?.version || 0
          updateVoiceStatus(payload.voice?.status)

          if (payload.comboEventVersion && payload.comboEventVersion !== lastComboEventVersion) {
            lastComboEventVersion = payload.comboEventVersion
            showComboBurst(payload.comboEventCount || 1, payload.comboEventLabel || 'Exercise')
          }

          if (voiceEnabled && payload.voice?.available && payload.voice?.version && payload.voice.version !== lastVoiceVersion) {
            if (!audioEl.paused && !audioEl.ended) {
              queuedVoiceVersion = payload.voice.version
            } else {
              queuedVoiceVersion = 0
              await playVoiceVersion(payload.voice.version)
            }
          }

          const joints = payload.joints || {}
          jointListEl.innerHTML = Object.keys(joints).length
            ? Object.entries(joints).map(([name, joint]) => renderJoint(name, joint)).join('')
            : '<div class="joint muted">No pose detected yet. Step into frame and wait a moment.</div>'
        } catch (error) {
          statusEl.className = 'status error'
          statusEl.innerHTML = '<span class="dot"></span><span>Failed to reach Python server</span>'
        }
      }

      exerciseSelectEl.addEventListener('change', async () => {
        try {
          await fetch('/exercise', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exercise: exerciseSelectEl.value }),
          })
        } catch (error) {
        }
      })

      sessionDurationSelectEl.addEventListener('change', async () => {
        try {
          await fetch('/session/duration', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ duration: sessionDurationSelectEl.value }),
          })
        } catch (error) {
        }
      })

      sessionButtonEl.addEventListener('click', async () => {
        try {
          await fetch('/session/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ duration: sessionDurationSelectEl.value, exercise: exerciseSelectEl.value }),
          })
        } catch (error) {
        }
      })

      coachButtonEl.addEventListener('click', async () => {
        try {
          await fetch('/rep-check/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exercise: exerciseSelectEl.value }),
          })
        } catch (error) {
        }
      })

      poll()
      setInterval(poll, 250)
    </script>
  </body>
</html>
"""


def draw_landmarks_on_frame(frame: Any, result: Any) -> Any:
    if not result or not result.pose_landmarks:
        return frame

    height, width, _ = frame.shape

    for pose in result.pose_landmarks:
        connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

        for connection in connections:
            start = pose[connection.start]
            end = pose[connection.end]

            x1, y1 = int(start.x * width), int(start.y * height)
            x2, y2 = int(end.x * width), int(end.y * height)

            cv2.line(frame, (x1, y1), (x2, y2), (92, 249, 170), 2)

        for landmark in pose:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 3, (255, 248, 242), -1)

    return frame


def landmark_visible(landmark: Any, threshold: float = VISIBILITY_THRESHOLD) -> bool:
    if landmark is None:
        return False
    visibility = landmark.visibility if landmark.visibility is not None else 1.0
    return visibility >= threshold


def calculate_angle(first: Any, mid: Any, end: Any) -> float:
    radians = math.atan2(end.y - mid.y, end.x - mid.x) - math.atan2(first.y - mid.y, first.x - mid.x)
    angle = abs(math.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def angle_from_vertical(lower: Any, upper: Any) -> float:
    dx = upper.x - lower.x
    dy = upper.y - lower.y
    return abs(math.degrees(math.atan2(dx, -dy)))


def get_side_score(pose: list[Any], indices: tuple[int, ...]) -> float:
    score = 0.0
    for index in indices:
        landmark = pose[index]
        score += landmark.visibility if landmark.visibility is not None else 1.0
    return score


def get_primary_side(pose: list[Any], indices: tuple[int, ...]) -> str:
    left_indices = tuple(index for index in indices if 'left' in INDEX_TO_NAME[index])
    right_indices = tuple(index for index in indices if 'right' in INDEX_TO_NAME[index])
    left_score = get_side_score(pose, left_indices)
    right_score = get_side_score(pose, right_indices)
    return 'left' if left_score >= right_score else 'right'


def get_pose_landmarks(result: Any) -> list[Any] | None:
    if not result or not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]


def stage_from_angle(angle: float, bottom_threshold: float, top_threshold: float) -> str:
    if angle <= bottom_threshold:
        return 'bottom'
    if angle >= top_threshold:
        return 'top'
    return 'mid'


def no_pose_analysis(selected_exercise: str = DEFAULT_EXERCISE) -> dict[str, Any]:
    label = exercise_label(selected_exercise)
    return {
        'exercise': selected_exercise,
        'exerciseLabel': label,
        'status': f'Waiting for a clear {label.lower()} pose',
        'metrics': [],
        'stage': 'unknown',
        'feedback': {
            'tone': 'neutral',
            'title': 'Need a clearer view',
            'details': [
                'Step farther back so your full body is visible.',
                'Keep shoulders, hips, knees, and ankles inside the frame.',
            ],
        },
    }


def partial_pose_analysis(selected_exercise: str) -> dict[str, Any]:
    label = exercise_label(selected_exercise)
    return {
        'exercise': selected_exercise,
        'exerciseLabel': label,
        'status': 'Pose found, but not enough visible joints for scoring',
        'metrics': [],
        'stage': 'unknown',
        'feedback': {
            'tone': 'neutral',
            'title': 'Need a clearer angle',
            'details': [
                'Turn slightly to the side for better joint angles.',
                'Keep the full moving side of your body visible.',
            ],
        },
    }


def tone_rank(tone: str) -> int:
    return {
        'neutral': 0,
        'good': 1,
        'warn': 2,
        'bad': 3,
    }.get(tone, 0)


def average_metric_sets(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for analysis in analyses:
        for metric in analysis.get('metrics', []):
            label = metric['label']
            if label not in grouped:
                grouped[label] = {
                    'label': label,
                    'target': metric.get('target', ''),
                    'value_total': 0.0,
                    'count': 0,
                }
            grouped[label]['value_total'] += float(metric.get('value', 0.0))
            grouped[label]['count'] += 1

    averaged: list[dict[str, Any]] = []
    for item in grouped.values():
        if item['count'] == 0:
            continue
        averaged.append({
            'label': item['label'],
            'value': item['value_total'] / item['count'],
            'target': item['target'],
        })
    return averaged


def aggregate_analysis_window(
    samples: list[dict[str, Any]],
    selected_exercise: str,
    window_started_at: int,
    window_ended_at: int,
) -> dict[str, Any]:
    if not samples:
        analysis = no_pose_analysis(selected_exercise)
        analysis['status'] = 'No form data captured in the last 5 seconds'
        analysis['feedback']['title'] = 'Waiting for movement in frame'
        analysis['windowStartedAt'] = window_started_at
        analysis['windowEndedAt'] = window_ended_at
        return analysis

    selected_analyses = [
        sample['analysis']
        for sample in samples
        if sample['analysis'].get('exercise') == selected_exercise
    ]
    if not selected_analyses:
        selected_analyses = [sample['analysis'] for sample in samples]

    base = dict(selected_analyses[-1])
    metric_summary = average_metric_sets(selected_analyses)
    detail_counts = Counter()
    title_counts = Counter()
    stage_counts = Counter()
    tone = 'neutral'

    for analysis in selected_analyses:
        feedback = analysis.get('feedback', {})
        current_tone = feedback.get('tone', 'neutral')
        if tone_rank(current_tone) > tone_rank(tone):
            tone = current_tone
        if feedback.get('title'):
            title_counts[feedback['title']] += 1
        for detail in feedback.get('details', []):
            detail_counts[detail] += 1
        if analysis.get('stage'):
            stage_counts[analysis['stage']] += 1

    details = [detail for detail, _ in detail_counts.most_common(3)]
    if not details:
        details = ['Keep moving for another 5 seconds so the app can gather more form data.']

    exercise_label = base.get('exerciseLabel', 'Exercise')
    status = f'{exercise_label} summary from the last 5 seconds'

    base['status'] = status
    base['metrics'] = metric_summary
    base['stage'] = stage_counts.most_common(1)[0][0] if stage_counts else base.get('stage', 'unknown')
    base['feedback'] = {
        'tone': tone,
        'title': title_counts.most_common(1)[0][0] if title_counts else base.get('feedback', {}).get('title', 'Form summary'),
        'details': details,
    }
    base['windowStartedAt'] = window_started_at
    base['windowEndedAt'] = window_ended_at
    return base


def build_spoken_feedback(analysis: dict[str, Any]) -> str | None:
    feedback = analysis.get('feedback', {})
    tone = feedback.get('tone', 'neutral')
    details = feedback.get('details', [])
    title = feedback.get('title', '')
    label = analysis.get('exerciseLabel', 'exercise')

    if tone in {'warn', 'bad'} and details:
        opener = random.choice(BAD_FORM_LINES)
        return f"{opener} {label} check. {title}. {details[0]}"

    if tone == 'good' and details:
        opener = random.choice(GOOD_FORM_LINES)
        return f"{opener} {label} check. {details[0]}"

    return None


def aggregate_rep_samples(samples: list[dict[str, Any]], selected_exercise: str) -> dict[str, Any]:
    if not samples:
        analysis = no_pose_analysis(selected_exercise)
        analysis['status'] = 'No scored rep was captured'
        analysis['feedback'] = {
            'tone': 'neutral',
            'title': 'Rep not captured',
            'details': [
                f'Start the {exercise_label(selected_exercise).lower()} from a clear position and complete one full rep.',
            ],
        }
        return analysis

    base = dict(samples[-1])
    metric_summary = average_metric_sets(samples)
    detail_counts = Counter()
    title_counts = Counter()
    stage_counts = Counter()
    tone = 'neutral'

    for analysis in samples:
        feedback = analysis.get('feedback', {})
        current_tone = feedback.get('tone', 'neutral')
        if tone_rank(current_tone) > tone_rank(tone):
            tone = current_tone
        if feedback.get('title'):
            title_counts[feedback['title']] += 1
        for detail in feedback.get('details', []):
            detail_counts[detail] += 1
        if analysis.get('stage'):
            stage_counts[analysis['stage']] += 1

    details = [detail for detail, _ in detail_counts.most_common(3)]
    if not details:
        details = ['Rep captured. Read the notes, then start another one-rep check when you are ready.']

    base['status'] = f"{base.get('exerciseLabel', exercise_label(selected_exercise))} rep captured and locked"
    base['metrics'] = metric_summary
    base['stage'] = stage_counts.most_common(1)[0][0] if stage_counts else base.get('stage', 'unknown')
    base['feedback'] = {
        'tone': tone,
        'title': title_counts.most_common(1)[0][0] if title_counts else 'Rep feedback locked',
        'details': details,
    }
    return base


class SessionTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.selected_duration_key = '60'
        self.high_scores = {key: 0 for key in SESSION_DURATION_OPTIONS}
        self.active = False
        self.started_at = 0
        self.ends_at = 0
        self.combo = 0
        self.best_combo = 0
        self.last_good_rep_at = 0
        self.last_stage = 'unknown'
        self.seen_bottom = False
        self.event_version = 0
        self.event_combo = 0
        self.event_label = ''

    def snapshot(self, now_ms: int | None = None) -> dict[str, Any]:
        with self.lock:
            remaining_ms = 0
            if self.active:
                current_time = now_ms or int(time.time() * 1000)
                remaining_ms = max(0, self.ends_at - current_time)

            return {
                'sessionActive': self.active,
                'sessionDurationKey': self.selected_duration_key,
                'sessionDurationLabel': session_duration_label(self.selected_duration_key),
                'sessionDurationOptions': SESSION_DURATION_SELECT_OPTIONS,
                'sessionRemainingMs': remaining_ms,
                'comboCount': self.combo,
                'comboBestCurrent': self.best_combo,
                'comboHighScores': dict(self.high_scores),
                'comboEventVersion': self.event_version,
                'comboEventCount': self.event_combo,
                'comboEventLabel': self.event_label,
            }

    def select_duration(self, duration_key: str) -> bool:
        if duration_key not in SESSION_DURATION_OPTIONS:
            return False

        with self.lock:
            self.selected_duration_key = duration_key
        return True

    def start_session(self, duration_key: str, now_ms: int) -> bool:
        if duration_key not in SESSION_DURATION_OPTIONS:
            return False

        with self.lock:
            duration_seconds = SESSION_DURATION_OPTIONS[duration_key]['seconds']
            self.selected_duration_key = duration_key
            self.active = True
            self.started_at = now_ms
            self.ends_at = now_ms + (duration_seconds * 1000)
            self.combo = 0
            self.best_combo = 0
            self.last_good_rep_at = 0
            self.last_stage = 'unknown'
            self.seen_bottom = False
            self.event_combo = 0
            self.event_label = ''
        return True

    def update(self, analysis: dict[str, Any], now_ms: int) -> None:
        with self.lock:
            if not self.active:
                return

            if now_ms >= self.ends_at:
                self._finish_session_locked()
                return

            if self.combo > 0 and self.last_good_rep_at and (now_ms - self.last_good_rep_at) > COMBO_WINDOW_MS:
                self.combo = 0

            stage = analysis.get('stage', 'unknown')
            if stage == 'bottom':
                self.seen_bottom = True

            if stage == 'top' and self.last_stage != 'top' and self.seen_bottom:
                self.seen_bottom = False
                if analysis.get('feedback', {}).get('tone') == 'good':
                    if self.last_good_rep_at and (now_ms - self.last_good_rep_at) <= COMBO_WINDOW_MS:
                        self.combo += 1
                    else:
                        self.combo = 1

                    self.last_good_rep_at = now_ms
                    self.best_combo = max(self.best_combo, self.combo)
                    duration_key = self.selected_duration_key
                    self.high_scores[duration_key] = max(self.high_scores[duration_key], self.best_combo)
                    self.event_version += 1
                    self.event_combo = self.combo
                    self.event_label = analysis.get('exerciseLabel', 'Exercise')
                else:
                    self.combo = 0

            self.last_stage = stage

            if now_ms >= self.ends_at:
                self._finish_session_locked()

    def _finish_session_locked(self) -> None:
        self.active = False
        self.combo = 0
        self.last_good_rep_at = 0
        self.last_stage = 'unknown'
        self.seen_bottom = False


class WorkoutTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.selected_exercise = DEFAULT_EXERCISE
        self.rep_count = 0
        self.coach_state = 'idle'
        self.last_stage = 'unknown'
        self.seen_bottom = False
        self.completed_analysis: dict[str, Any] | None = None
        self.current_rep_samples: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            exercise = self.selected_exercise
            coach_title = 'Pick an exercise, then start a one-rep check.'
            coach_message = build_rep_prompt(exercise)
            coach_button_label = 'Start one-rep check'
            coach_button_disabled = False

            if self.coach_state == 'waiting':
                coach_title = 'Ready. Do one clean rep now.'
                coach_message = f"I am waiting for one full {exercise_label(exercise).lower()} rep. I will lock the feedback as soon as you finish it."
                coach_button_label = 'Watching your rep...'
                coach_button_disabled = True
            elif self.coach_state == 'collecting':
                coach_title = 'Rep in progress.'
                coach_message = 'Finish this rep normally. Feedback will stay locked after this rep completes.'
                coach_button_label = 'Watching your rep...'
                coach_button_disabled = True
            elif self.coach_state == 'complete':
                coach_title = 'Feedback locked for your last rep.'
                coach_message = 'Read the notes below. When you want another score, press the button to analyze one new rep.'
                coach_button_label = 'Analyze next rep'

            return {
                'selectedExercise': exercise,
                'selectedExerciseLabel': exercise_label(exercise),
                'supportedExercises': SUPPORTED_EXERCISE_OPTIONS,
                'repCount': self.rep_count,
                'coachState': self.coach_state,
                'coachTitle': coach_title,
                'coachMessage': coach_message,
                'coachButtonLabel': coach_button_label,
                'coachButtonDisabled': coach_button_disabled,
                'analysisLocked': self.coach_state == 'complete',
            }

    def current_exercise(self) -> str:
        with self.lock:
            return self.selected_exercise

    def select_exercise(self, exercise: str) -> bool:
        if exercise not in SUPPORTED_EXERCISES:
            return False

        with self.lock:
            self.selected_exercise = exercise
            self.rep_count = 0
            self.coach_state = 'idle'
            self.last_stage = 'unknown'
            self.seen_bottom = False
            self.completed_analysis = None
            self.current_rep_samples = []
        return True

    def start_rep_check(self) -> str:
        with self.lock:
            self.coach_state = 'waiting'
            self.last_stage = 'unknown'
            self.seen_bottom = False
            self.completed_analysis = None
            self.current_rep_samples = []
            return build_rep_prompt(self.selected_exercise)

    def displayed_analysis(self) -> dict[str, Any] | None:
        with self.lock:
            if self.coach_state == 'complete' and self.completed_analysis is not None:
                return dict(self.completed_analysis)
            return None

    def update(self, analysis: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        with self.lock:
            if self.coach_state not in {'waiting', 'collecting'}:
                return self.completed_analysis, None

            stage = analysis.get('stage', 'unknown')
            if stage == 'unknown':
                return self.completed_analysis, None

            if self.coach_state == 'waiting':
                self.coach_state = 'collecting'

            self.current_rep_samples.append(analysis)

            if stage == 'bottom':
                self.seen_bottom = True

            locked_analysis: dict[str, Any] | None = None
            voice_text: str | None = None
            if stage == 'top' and self.last_stage != 'top' and self.seen_bottom:
                self.rep_count += 1
                self.seen_bottom = False
                self.coach_state = 'complete'
                locked_analysis = aggregate_rep_samples(self.current_rep_samples, self.selected_exercise)
                self.completed_analysis = locked_analysis
                self.current_rep_samples = []
                voice_text = build_spoken_feedback(locked_analysis)
                if voice_text is None:
                    voice_text = f"{exercise_label(self.selected_exercise)} rep captured. Feedback is now locked on screen."

            self.last_stage = stage
            return locked_analysis, voice_text


def analyze_squat(pose: list[Any]) -> dict[str, Any]:
    side = get_primary_side(pose, (11, 12, 23, 24, 25, 26, 27, 28))
    shoulder = pose[11] if side == 'left' else pose[12]
    hip = pose[23] if side == 'left' else pose[24]
    knee = pose[25] if side == 'left' else pose[26]
    ankle = pose[27] if side == 'left' else pose[28]

    if not all(landmark_visible(item) for item in (shoulder, hip, knee, ankle)):
        return partial_pose_analysis('squat')

    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle = calculate_angle(shoulder, hip, knee)
    torso_lean = angle_from_vertical(hip, shoulder)
    stage = stage_from_angle(knee_angle, 105.0, 155.0)
    tone = 'good'
    title = 'Strong squat position'
    details: list[str] = []

    if stage != 'top' and knee_angle > 118.0:
        tone = 'warn'
        title = 'Needs more depth'
        details.append('Go a little deeper so your knees bend more at the bottom.')

    if torso_lean > 36.0:
        tone = 'bad' if tone == 'warn' else 'warn'
        title = 'Chest is dropping forward'
        details.append('Keep your chest taller and brace your core to avoid folding forward.')

    if hip_angle < 55.0:
        tone = 'bad' if tone == 'warn' else 'warn'
        title = 'Hips are collapsing too much'
        details.append('Sit back with control and keep tension through your hips.')

    if not details:
        if stage == 'top':
            tone = 'neutral'
            title = 'Ready for the next squat'
            details.append('Sit your hips back, bend your knees, then drive back up tall.')
        elif stage == 'bottom':
            details.append('Depth looks solid. Drive through your feet and stand tall.')
        else:
            details.append('Good control. Keep knees and hips moving together.')

    return {
        'exercise': 'squat',
        'exerciseLabel': 'Squat',
        'stage': stage,
        'status': f'Squat detected on {side} side',
        'metrics': [
            {'label': 'Knee angle', 'value': knee_angle, 'target': '90-110 deg bottom'},
            {'label': 'Hip angle', 'value': hip_angle, 'target': 'stay controlled'},
            {'label': 'Torso lean', 'value': torso_lean, 'target': 'under 35 deg'},
        ],
        'feedback': {'tone': tone, 'title': title, 'details': details},
    }


def analyze_pushup(pose: list[Any]) -> dict[str, Any]:
    side = get_primary_side(pose, (11, 12, 13, 14, 15, 16, 23, 24, 27, 28))
    shoulder = pose[11] if side == 'left' else pose[12]
    elbow = pose[13] if side == 'left' else pose[14]
    wrist = pose[15] if side == 'left' else pose[16]
    hip = pose[23] if side == 'left' else pose[24]
    ankle = pose[27] if side == 'left' else pose[28]

    if not all(landmark_visible(item) for item in (shoulder, elbow, wrist, hip, ankle)):
        return partial_pose_analysis('pushup')

    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    body_line = calculate_angle(shoulder, hip, ankle)
    stage = stage_from_angle(elbow_angle, 92.0, 152.0)
    tone = 'good'
    title = 'Pushup line looks strong'
    details: list[str] = []

    if stage != 'top' and elbow_angle > 105.0:
        tone = 'warn'
        title = 'Lower for more depth'
        details.append('Lower a bit more so your elbows bend further at the bottom.')

    if body_line < 155.0:
        tone = 'bad' if tone == 'warn' else 'warn'
        title = 'Core line is breaking'
        details.append('Keep your body in one straight line from shoulders through ankles.')

    if not details:
        if stage == 'top':
            tone = 'neutral'
            title = 'Ready for the next pushup'
            details.append('Brace your core, bend your elbows, and lower under control.')
        elif stage == 'bottom':
            details.append('Nice depth. Press back up without losing your body line.')
        else:
            details.append('Good control. Keep your torso and legs moving as one unit.')

    return {
        'exercise': 'pushup',
        'exerciseLabel': 'Pushup',
        'stage': stage,
        'status': f'Pushup detected on {side} side',
        'metrics': [
            {'label': 'Elbow angle', 'value': elbow_angle, 'target': '80-95 deg bottom'},
            {'label': 'Body line', 'value': body_line, 'target': '155-180 deg'},
        ],
        'feedback': {'tone': tone, 'title': title, 'details': details},
    }


def analyze_bicep_curl(pose: list[Any]) -> dict[str, Any]:
    side = get_primary_side(pose, (11, 12, 13, 14, 15, 16, 23, 24))
    shoulder = pose[11] if side == 'left' else pose[12]
    elbow = pose[13] if side == 'left' else pose[14]
    wrist = pose[15] if side == 'left' else pose[16]
    hip = pose[23] if side == 'left' else pose[24]

    if not all(landmark_visible(item) for item in (shoulder, elbow, wrist, hip)):
        return partial_pose_analysis('bicep_curl')

    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    upper_arm_drift = calculate_angle(hip, shoulder, elbow)
    stage = stage_from_angle(elbow_angle, 70.0, 145.0)
    tone = 'good'
    title = 'Curl mechanics look strong'
    details: list[str] = []

    if stage != 'top' and elbow_angle > 88.0:
        tone = 'warn'
        title = 'Finish the curl higher'
        details.append('Bring the weight higher so the elbow closes more at the top.')

    if upper_arm_drift > 28.0:
        tone = 'bad' if tone == 'warn' else 'warn'
        title = 'Upper arm is swinging'
        details.append('Keep your elbow tucked closer to your side and avoid shoulder swing.')

    if not details:
        if stage == 'top':
            details.append('Strong squeeze at the top. Lower back down under control.')
        elif stage == 'mid':
            details.append('Good curl path. Keep the elbow pinned while the forearm moves.')
        else:
            tone = 'neutral'
            title = 'Ready for the next curl'
            details.append('Start tall, keep the elbow tucked, and curl without swinging.')

    return {
        'exercise': 'bicep_curl',
        'exerciseLabel': 'Bicep curl',
        'stage': stage,
        'status': f'Bicep curl detected on {side} side',
        'metrics': [
            {'label': 'Elbow angle', 'value': elbow_angle, 'target': '55-75 deg top'},
            {'label': 'Upper arm drift', 'value': upper_arm_drift, 'target': 'under 25 deg'},
        ],
        'feedback': {'tone': tone, 'title': title, 'details': details},
    }


def analyze_overhead_press(pose: list[Any]) -> dict[str, Any]:
    side = get_primary_side(pose, (11, 12, 13, 14, 15, 16, 23, 24))
    shoulder = pose[11] if side == 'left' else pose[12]
    elbow = pose[13] if side == 'left' else pose[14]
    wrist = pose[15] if side == 'left' else pose[16]
    hip = pose[23] if side == 'left' else pose[24]

    if not all(landmark_visible(item) for item in (shoulder, elbow, wrist, hip)):
        return partial_pose_analysis('overhead_press')

    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    arm_stack = angle_from_vertical(shoulder, wrist)
    stage = 'top' if wrist.y < shoulder.y and elbow_angle >= 145.0 else 'bottom' if wrist.y > shoulder.y else 'mid'
    tone = 'good'
    title = 'Press path looks solid'
    details: list[str] = []

    if stage == 'top' and elbow_angle < 155.0:
        tone = 'warn'
        title = 'Finish taller overhead'
        details.append('Reach higher at the top so the press fully locks out.')

    if arm_stack > 26.0:
        tone = 'bad' if tone == 'warn' else 'warn'
        title = 'Stack the arm more vertically'
        details.append('Keep your wrist closer over the shoulder instead of drifting forward.')

    if not details:
        if stage == 'top':
            details.append('Nice lockout. Lower with control and press straight back up.')
        elif stage == 'mid':
            details.append('Smooth press. Keep the wrist stacked over the shoulder.')
        else:
            tone = 'neutral'
            title = 'Ready for the next press'
            details.append('Start at shoulder level, brace, and drive the weight overhead.')

    return {
        'exercise': 'overhead_press',
        'exerciseLabel': 'Overhead press',
        'stage': stage,
        'status': f'Overhead press detected on {side} side',
        'metrics': [
            {'label': 'Elbow angle', 'value': elbow_angle, 'target': 'near 160 deg at top'},
            {'label': 'Arm stack', 'value': arm_stack, 'target': 'under 25 deg'},
        ],
        'feedback': {'tone': tone, 'title': title, 'details': details},
    }


def analyze_situp(pose: list[Any]) -> dict[str, Any]:
    side = get_primary_side(pose, (11, 12, 23, 24, 25, 26, 27, 28))
    shoulder = pose[11] if side == 'left' else pose[12]
    hip = pose[23] if side == 'left' else pose[24]
    knee = pose[25] if side == 'left' else pose[26]
    ankle = pose[27] if side == 'left' else pose[28]

    if not all(landmark_visible(item) for item in (shoulder, hip, knee, ankle)):
        return partial_pose_analysis('situp')

    torso_angle = angle_from_vertical(hip, shoulder)
    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)
    stage = 'top' if torso_angle < 28.0 else 'bottom' if torso_angle > 55.0 else 'mid'
    tone = 'good'
    title = 'Situp rhythm looks strong'
    details: list[str] = []

    if stage != 'top' and torso_angle > 62.0:
        tone = 'warn'
        title = 'Come up a little higher'
        details.append('Curl higher at the top so your torso gets more upright.')

    if knee_angle > 160.0:
        tone = 'bad' if tone == 'warn' else 'warn'
        title = 'Bend the knees more'
        details.append('Keep your knees bent so the situp position stays more stable.')

    if not details:
        if stage == 'top':
            details.append('Nice crunch. Lower under control and keep tension through your core.')
        elif stage == 'mid':
            details.append('Good tempo. Keep the movement driven by your abs, not momentum.')
        else:
            tone = 'neutral'
            title = 'Ready for the next situp'
            details.append('Brace your core, curl up smoothly, and lower without flopping back.')

    return {
        'exercise': 'situp',
        'exerciseLabel': 'Situp',
        'stage': stage,
        'status': f'Situp detected on {side} side',
        'metrics': [
            {'label': 'Torso angle', 'value': torso_angle, 'target': 'under 30 deg at top'},
            {'label': 'Hip angle', 'value': hip_angle, 'target': 'stay controlled'},
        ],
        'feedback': {'tone': tone, 'title': title, 'details': details},
    }


def analyze_lunge(pose: list[Any]) -> dict[str, Any]:
    side = get_primary_side(pose, (11, 12, 23, 24, 25, 26, 27, 28))
    shoulder = pose[11] if side == 'left' else pose[12]
    hip = pose[23] if side == 'left' else pose[24]
    knee = pose[25] if side == 'left' else pose[26]
    ankle = pose[27] if side == 'left' else pose[28]

    if not all(landmark_visible(item) for item in (shoulder, hip, knee, ankle)):
        return partial_pose_analysis('lunge')

    knee_angle = calculate_angle(hip, knee, ankle)
    torso_lean = angle_from_vertical(hip, shoulder)
    hip_angle = calculate_angle(shoulder, hip, knee)
    stage = stage_from_angle(knee_angle, 112.0, 158.0)
    tone = 'good'
    title = 'Lunge position looks strong'
    details: list[str] = []

    if stage != 'top' and knee_angle > 122.0:
        tone = 'warn'
        title = 'Drop deeper into the lunge'
        details.append('Lower a bit more so the front knee bends closer to ninety degrees.')

    if torso_lean > 24.0:
        tone = 'bad' if tone == 'warn' else 'warn'
        title = 'Stay more upright'
        details.append('Keep your chest taller instead of leaning forward into the rep.')

    if not details:
        if stage == 'top':
            tone = 'neutral'
            title = 'Ready for the next lunge'
            details.append('Step down with control, then drive back up tall through the front leg.')
        elif stage == 'bottom':
            details.append('Nice depth. Push through the floor and stand back up smoothly.')
        else:
            details.append('Good control. Keep the front knee tracking cleanly over the foot.')

    return {
        'exercise': 'lunge',
        'exerciseLabel': 'Lunge',
        'stage': stage,
        'status': f'Lunge detected on {side} side',
        'metrics': [
            {'label': 'Knee angle', 'value': knee_angle, 'target': '90-110 deg bottom'},
            {'label': 'Torso lean', 'value': torso_lean, 'target': 'under 25 deg'},
            {'label': 'Hip angle', 'value': hip_angle, 'target': 'stay controlled'},
        ],
        'feedback': {'tone': tone, 'title': title, 'details': details},
    }


def analyze_pose(result: Any, selected_exercise: str) -> dict[str, Any]:
    pose = get_pose_landmarks(result)
    if pose is None:
        return no_pose_analysis(selected_exercise)

    analyzers = {
        'squat': analyze_squat,
        'pushup': analyze_pushup,
        'bicep_curl': analyze_bicep_curl,
        'overhead_press': analyze_overhead_press,
        'situp': analyze_situp,
        'lunge': analyze_lunge,
    }
    analyzer = analyzers.get(selected_exercise)
    if analyzer is None:
        return no_pose_analysis(DEFAULT_EXERCISE)
    return analyzer(pose)


def build_joint_payload(result: Any) -> dict[str, Any]:
    if not result or not result.pose_landmarks:
        return {}

    pose = result.pose_landmarks[0]
    payload: dict[str, Any] = {}
    for index, name in POLL_NAMES.items():
        landmark = pose[index]
        payload[name] = {
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility if landmark.visibility is not None else 1.0,
        }
    return payload


class PoseRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.latest_payload: dict[str, Any] = {
            'status': 'Starting camera...',
            'error': '',
            'updatedAt': 0,
            'frameWidth': 0,
            'frameHeight': 0,
            'landmarkCount': 0,
            'joints': {},
            'analysis': None,
            'selectedExercise': DEFAULT_EXERCISE,
            'selectedExerciseLabel': exercise_label(DEFAULT_EXERCISE),
            'supportedExercises': SUPPORTED_EXERCISE_OPTIONS,
            'repCount': 0,
            'sessionActive': False,
            'sessionDurationKey': '60',
            'sessionDurationLabel': session_duration_label('60'),
            'sessionDurationOptions': SESSION_DURATION_SELECT_OPTIONS,
            'sessionRemainingMs': 0,
            'comboCount': 0,
            'comboBestCurrent': 0,
            'comboHighScores': {key: 0 for key in SESSION_DURATION_OPTIONS},
            'comboEventVersion': 0,
            'comboEventCount': 0,
            'comboEventLabel': '',
            'voice': {
                'available': False,
                'status': 'Set ELEVENLABS_API_KEY to enable voice coaching.',
                'version': 0,
            },
        }
        self.running = True

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.latest_payload)

    def frame(self) -> bytes | None:
        with self.lock:
            return self.latest_jpeg

    def update(self, jpeg: bytes, payload: dict[str, Any]) -> None:
        with self.lock:
            self.latest_jpeg = jpeg
            self.latest_payload = payload

    def update_payload(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.latest_payload = payload

    def set_error(self, message: str) -> None:
        with self.lock:
            self.latest_payload = {
                'status': 'Camera error',
                'error': message,
                'updatedAt': int(time.time() * 1000),
                'frameWidth': 0,
                'frameHeight': 0,
                'landmarkCount': 0,
                'joints': {},
                'analysis': no_pose_analysis(self.latest_payload.get('selectedExercise', DEFAULT_EXERCISE)),
                'selectedExercise': self.latest_payload.get('selectedExercise', DEFAULT_EXERCISE),
                'selectedExerciseLabel': self.latest_payload.get('selectedExerciseLabel', exercise_label(DEFAULT_EXERCISE)),
                'supportedExercises': self.latest_payload.get('supportedExercises', SUPPORTED_EXERCISE_OPTIONS),
                'repCount': self.latest_payload.get('repCount', 0),
                'sessionActive': self.latest_payload.get('sessionActive', False),
                'sessionDurationKey': self.latest_payload.get('sessionDurationKey', '60'),
                'sessionDurationLabel': self.latest_payload.get('sessionDurationLabel', session_duration_label('60')),
                'sessionDurationOptions': self.latest_payload.get('sessionDurationOptions', SESSION_DURATION_SELECT_OPTIONS),
                'sessionRemainingMs': self.latest_payload.get('sessionRemainingMs', 0),
                'comboCount': self.latest_payload.get('comboCount', 0),
                'comboBestCurrent': self.latest_payload.get('comboBestCurrent', 0),
                'comboHighScores': self.latest_payload.get('comboHighScores', {key: 0 for key in SESSION_DURATION_OPTIONS}),
                'comboEventVersion': self.latest_payload.get('comboEventVersion', 0),
                'comboEventCount': self.latest_payload.get('comboEventCount', 0),
                'comboEventLabel': self.latest_payload.get('comboEventLabel', ''),
                'voice': self.latest_payload.get('voice', {
                    'available': False,
                    'status': 'Set ELEVENLABS_API_KEY to enable voice coaching.',
                    'version': 0,
                }),
            }


class VoiceManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.api_key = os.getenv('ELEVENLABS_API_KEY', '')
        self.client = ElevenLabs(api_key=self.api_key) if self.api_key else None
        self.audio_bytes: bytes = b''
        self.version = 0
        self.status = 'Voice ready.' if self.client else 'Set ELEVENLABS_API_KEY to enable voice coaching.'
        self.last_text = ''
        self.is_generating = False

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                'available': self.client is not None,
                'status': self.status,
                'version': self.version,
            }

    def audio(self) -> bytes:
        with self.lock:
            return self.audio_bytes

    def request(self, text: str | None) -> None:
        if not text or self.client is None:
            return

        with self.lock:
            if self.is_generating or text == self.last_text:
                return
            self.is_generating = True
            self.last_text = text
            self.status = 'Generating voice coaching...'

        worker = threading.Thread(target=self._generate, args=(text,), daemon=True)
        worker.start()

    def _generate(self, text: str) -> None:
        try:
            audio_chunks = self.client.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                text=text,
                model_id=ELEVENLABS_MODEL_ID,
                output_format='mp3_44100_128',
            )
            audio = b''.join(audio_chunks)
            with self.lock:
                self.audio_bytes = audio
                self.version += 1
                self.status = 'Voice coaching ready.'
        except Exception as exc:
            with self.lock:
                self.status = f'Voice generation failed: {exc}'
                self.last_text = ''
        finally:
            with self.lock:
                self.is_generating = False


runtime = PoseRuntime()
workout_tracker = WorkoutTracker()
session_tracker = SessionTracker()
voice_manager = VoiceManager()
runtime.latest_payload.update(workout_tracker.snapshot())
runtime.latest_payload.update(session_tracker.snapshot())
runtime.latest_payload['voice'] = voice_manager.snapshot()


def capture_loop() -> None:
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            cap, camera_error = open_camera()
            if cap is None:
                runtime.set_error(camera_error or 'Could not open webcam from Python.')
                return

            try:
                while runtime.running:
                    ok, frame = cap.read()
                    if not ok:
                        runtime.set_error('Failed to read a frame from the webcam.')
                        time.sleep(0.1)
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    selected_exercise = workout_tracker.current_exercise()
                    timestamp_ms = int(time.time() * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    frame = draw_landmarks_on_frame(frame, result)

                    encoded, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if not encoded:
                        continue

                    joints = build_joint_payload(result)
                    analysis = analyze_pose(result, selected_exercise)
                    locked_analysis, voice_text = workout_tracker.update(analysis)
                    session_tracker.update(analysis, timestamp_ms)
                    if voice_text:
                        voice_manager.request(voice_text)

                    published_analysis = locked_analysis or workout_tracker.displayed_analysis()

                    tracker_snapshot = workout_tracker.snapshot()
                    session_snapshot = session_tracker.snapshot(timestamp_ms)
                    payload = {
                        'status': 'Camera live' if joints else 'Searching for pose...',
                        'error': '',
                        'updatedAt': timestamp_ms,
                        'frameWidth': int(frame.shape[1]),
                        'frameHeight': int(frame.shape[0]),
                        'landmarkCount': len(result.pose_landmarks[0]) if result and result.pose_landmarks else 0,
                        'joints': joints,
                        'analysis': published_analysis,
                        **tracker_snapshot,
                        **session_snapshot,
                        'voice': voice_manager.snapshot(),
                    }
                    runtime.update(jpeg.tobytes(), payload)
            finally:
                cap.release()
    except Exception as exc:
        runtime.set_error(f'Pose runtime failed: {exc}')


class PoseRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == '/':
            self._write_html()
            return
        if self.path == '/pose.json':
            self._write_json(runtime.snapshot())
            return
        if self.path.startswith('/coach-audio.mp3'):
            self._write_audio()
            return
        if self.path == '/stream.mjpg':
            self._write_stream()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == '/exercise':
            self._update_exercise()
            return
        if self.path == '/rep-check/start':
            self._start_rep_check()
            return
        if self.path == '/session/duration':
            self._update_session_duration()
            return
        if self.path == '/session/start':
            self._start_session()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_html(self) -> None:
        body = HTML_PAGE.encode('utf-8')
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode('utf-8')
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _update_exercise(self) -> None:
        content_length = int(self.headers.get('Content-Length', '0'))
        raw_body = self.rfile.read(content_length) if content_length else b'{}'

        try:
            payload = json.loads(raw_body.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, 'Invalid JSON body')
            return

        exercise = payload.get('exercise')
        if not isinstance(exercise, str) or not workout_tracker.select_exercise(exercise):
            self.send_error(HTTPStatus.BAD_REQUEST, 'Unsupported exercise')
            return

        tracker_snapshot = workout_tracker.snapshot()
        latest = runtime.snapshot()
        latest.update(tracker_snapshot)
        latest.update(session_tracker.snapshot())
        latest['analysis'] = None
        runtime.update_payload(latest)
        self._write_json({'ok': True, **tracker_snapshot})

    def _start_rep_check(self) -> None:
        content_length = int(self.headers.get('Content-Length', '0'))
        raw_body = self.rfile.read(content_length) if content_length else b'{}'

        try:
            payload = json.loads(raw_body.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, 'Invalid JSON body')
            return

        exercise = payload.get('exercise')
        if isinstance(exercise, str) and exercise in SUPPORTED_EXERCISES:
            workout_tracker.select_exercise(exercise)

        prompt = workout_tracker.start_rep_check()
        voice_manager.request(prompt)

        tracker_snapshot = workout_tracker.snapshot()
        latest = runtime.snapshot()
        latest.update(tracker_snapshot)
        latest.update(session_tracker.snapshot())
        latest['analysis'] = None
        runtime.update_payload(latest)
        self._write_json({'ok': True, **tracker_snapshot})

    def _update_session_duration(self) -> None:
        content_length = int(self.headers.get('Content-Length', '0'))
        raw_body = self.rfile.read(content_length) if content_length else b'{}'

        try:
            payload = json.loads(raw_body.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, 'Invalid JSON body')
            return

        duration_key = payload.get('duration')
        if not isinstance(duration_key, str) or not session_tracker.select_duration(duration_key):
            self.send_error(HTTPStatus.BAD_REQUEST, 'Unsupported session duration')
            return

        latest = runtime.snapshot()
        latest.update(session_tracker.snapshot())
        runtime.update_payload(latest)
        self._write_json({'ok': True, **session_tracker.snapshot()})

    def _start_session(self) -> None:
        content_length = int(self.headers.get('Content-Length', '0'))
        raw_body = self.rfile.read(content_length) if content_length else b'{}'

        try:
            payload = json.loads(raw_body.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, 'Invalid JSON body')
            return

        exercise = payload.get('exercise')
        if isinstance(exercise, str) and exercise in SUPPORTED_EXERCISES:
            workout_tracker.select_exercise(exercise)

        duration_key = payload.get('duration')
        now_ms = int(time.time() * 1000)
        if not isinstance(duration_key, str) or not session_tracker.start_session(duration_key, now_ms):
            self.send_error(HTTPStatus.BAD_REQUEST, 'Unsupported session duration')
            return

        latest = runtime.snapshot()
        latest.update(workout_tracker.snapshot())
        latest.update(session_tracker.snapshot(now_ms))
        runtime.update_payload(latest)
        self._write_json({'ok': True, **session_tracker.snapshot(now_ms)})

    def _write_audio(self) -> None:
        body = voice_manager.audio()
        if not body:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'audio/mpeg')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Connection', 'close')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        try:
            while True:
                frame = runtime.frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                self.wfile.write(FRAME_BOUNDARY)
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    worker = threading.Thread(target=capture_loop, daemon=True)
    worker.start()

    server = ThreadingHTTPServer((HOST, PORT), PoseRequestHandler)
    print(f'Python pose server running at http://{HOST}:{PORT}')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.running = False
        server.server_close()


if __name__ == '__main__':
    main()
