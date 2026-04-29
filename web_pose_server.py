import json
import math
import os
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
ANALYSIS_WINDOW_MS = 5000
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', '21m00Tcm4TlvDq8ikWAM')
ELEVENLABS_MODEL_ID = os.getenv('ELEVENLABS_MODEL_ID', 'eleven_multilingual_v2')
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
        margin: 0 22px 22px;
        background: #02070d;
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
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
          <img src="/stream.mjpg" alt="Live pose stream" />
        </div>
      </section>

      <aside class="panel sidebar">
        <div id="status" class="status"><span class="dot"></span><span>Starting camera...</span></div>
        <div class="stats">
          <div class="stat"><span>Frame size</span><strong id="frame-size">-</strong></div>
          <div class="stat"><span>Last update</span><strong id="timestamp">-</strong></div>
          <div class="stat"><span>Pose landmarks</span><strong id="landmark-count">0</strong></div>
        </div>
        <div class="voice-row">
          <button id="voice-toggle" class="voice-button" type="button">Enable voice</button>
          <span id="voice-status" class="muted">Voice unavailable</span>
        </div>
        <section id="analysis" class="analysis neutral">
          <div class="muted">Waiting for exercise analysis...</div>
        </section>
        <div id="joint-list" class="joint-list"></div>
      </aside>
    </div>

    <script>
      const statusEl = document.getElementById('status')
      const frameSizeEl = document.getElementById('frame-size')
      const timestampEl = document.getElementById('timestamp')
      const landmarkCountEl = document.getElementById('landmark-count')
      const analysisEl = document.getElementById('analysis')
      const jointListEl = document.getElementById('joint-list')
      const voiceToggleEl = document.getElementById('voice-toggle')
      const voiceStatusEl = document.getElementById('voice-status')
      const audioEl = new Audio()
      let voiceEnabled = false
      let lastVoiceVersion = 0

      voiceToggleEl.addEventListener('click', async () => {
        voiceEnabled = !voiceEnabled
        voiceToggleEl.textContent = voiceEnabled ? 'Disable voice' : 'Enable voice'
        if (voiceEnabled) {
          try {
            audioEl.muted = false
            await audioEl.play().catch(() => {})
            audioEl.pause()
            audioEl.currentTime = 0
          } catch (error) {
          }
        }
      })

      function renderJoint(name, joint) {
        return `<div class="joint"><strong>${name}</strong><br><span class="muted">x ${joint.x.toFixed(3)} · y ${joint.y.toFixed(3)} · z ${joint.z.toFixed(3)} · vis ${Math.round(joint.visibility * 100)}%</span></div>`
      }

      function renderAnalysis(analysis) {
        if (!analysis) {
          return '<div class="muted">No exercise analysis yet.</div>'
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
          analysisEl.className = `analysis ${payload.analysis?.feedback?.tone || 'neutral'}`
          analysisEl.innerHTML = renderAnalysis(payload.analysis)
          voiceToggleEl.disabled = !payload.voice?.available
          voiceStatusEl.textContent = payload.voice?.status || 'Voice unavailable'

          if (voiceEnabled && payload.voice?.available && payload.voice?.version && payload.voice.version !== lastVoiceVersion) {
            lastVoiceVersion = payload.voice.version
            audioEl.src = `/coach-audio.mp3?v=${payload.voice.version}`
            try {
              await audioEl.play()
            } catch (error) {
              voiceStatusEl.textContent = 'Voice ready - browser blocked autoplay, click Enable voice again.'
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


def no_pose_analysis() -> dict[str, Any]:
    return {
        'exercise': 'unknown',
        'exerciseLabel': 'No exercise',
        'status': 'Waiting for a clear full-body pose',
        'metrics': [],
        'feedback': {
            'tone': 'neutral',
            'title': 'Need a clearer view',
            'details': [
                'Step farther back so your full body is visible.',
                'Keep shoulders, hips, knees, and ankles inside the frame.',
            ],
        },
    }


def partial_pose_analysis(label: str) -> dict[str, Any]:
    return {
        'exercise': 'unknown',
        'exerciseLabel': label,
        'status': 'Pose found, but not enough visible joints for scoring',
        'metrics': [],
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


def aggregate_analysis_window(samples: list[dict[str, Any]], window_started_at: int, window_ended_at: int) -> dict[str, Any]:
    if not samples:
        analysis = no_pose_analysis()
        analysis['status'] = 'No form data captured in the last 5 seconds'
        analysis['feedback']['title'] = 'Waiting for movement in frame'
        analysis['windowStartedAt'] = window_started_at
        analysis['windowEndedAt'] = window_ended_at
        return analysis

    exercise_counts = Counter(
        sample['analysis'].get('exercise', 'unknown')
        for sample in samples
        if sample['analysis'].get('exercise', 'unknown') != 'unknown'
    )
    dominant_exercise = exercise_counts.most_common(1)[0][0] if exercise_counts else 'unknown'

    selected_analyses = [
        sample['analysis']
        for sample in samples
        if sample['analysis'].get('exercise', 'unknown') == dominant_exercise
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
    if dominant_exercise == 'unknown':
        status = 'Summary from the last 5 seconds'
    else:
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
    exercise = analysis.get('exercise', 'unknown')
    feedback = analysis.get('feedback', {})
    tone = feedback.get('tone', 'neutral')
    details = feedback.get('details', [])
    title = feedback.get('title', '')
    label = analysis.get('exerciseLabel', 'exercise')

    if exercise == 'unknown':
        return None

    if tone in {'warn', 'bad'} and details:
        return f"{label} check. {title}. {details[0]}"

    if tone == 'good' and details:
        return f"Good {label.lower()} form. {details[0]}"

    return None


def analyze_squat(pose: list[Any]) -> dict[str, Any]:
    side = get_primary_side(pose, (11, 12, 23, 24, 25, 26, 27, 28))
    shoulder = pose[11] if side == 'left' else pose[12]
    hip = pose[23] if side == 'left' else pose[24]
    knee = pose[25] if side == 'left' else pose[26]
    ankle = pose[27] if side == 'left' else pose[28]

    if not all(landmark_visible(item) for item in (shoulder, hip, knee, ankle)):
        return partial_pose_analysis('Squat')

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
        return partial_pose_analysis('Pushup')

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
        return partial_pose_analysis('Bicep curl')

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


def analyze_pose(result: Any) -> dict[str, Any]:
    pose = get_pose_landmarks(result)
    if pose is None:
        return no_pose_analysis()

    left_shoulder = pose[11]
    right_shoulder = pose[12]
    left_hip = pose[23]
    right_hip = pose[24]
    left_ankle = pose[27]
    right_ankle = pose[28]
    left_elbow = pose[13]
    right_elbow = pose[14]
    left_knee = pose[25]
    right_knee = pose[26]

    torso_reference_side = 'left' if (left_shoulder.visibility or 0) >= (right_shoulder.visibility or 0) else 'right'
    shoulder = left_shoulder if torso_reference_side == 'left' else right_shoulder
    hip = left_hip if torso_reference_side == 'left' else right_hip
    ankle = left_ankle if torso_reference_side == 'left' else right_ankle
    torso_angle = angle_from_vertical(hip, shoulder) if all(landmark_visible(item, 0.2) for item in (shoulder, hip)) else 0.0
    min_elbow_angle = min(calculate_angle(left_shoulder, left_elbow, pose[15]), calculate_angle(right_shoulder, right_elbow, pose[16]))
    min_knee_angle = min(calculate_angle(left_hip, left_knee, left_ankle), calculate_angle(right_hip, right_knee, right_ankle))
    body_line = calculate_angle(shoulder, hip, ankle) if all(landmark_visible(item, 0.2) for item in (shoulder, hip, ankle)) else 0.0

    if torso_angle > 55.0 and body_line > 140.0:
        return analyze_pushup(pose)
    if min_knee_angle < 150.0:
        return analyze_squat(pose)
    if min_elbow_angle < 135.0:
        return analyze_bicep_curl(pose)

    return {
        'exercise': 'unknown',
        'exerciseLabel': 'No exercise locked',
        'status': 'Standing pose detected',
        'metrics': [
            {'label': 'Torso angle', 'value': torso_angle, 'target': 'side view helps'},
            {'label': 'Min knee angle', 'value': min_knee_angle, 'target': 'drops in squats'},
            {'label': 'Min elbow angle', 'value': min_elbow_angle, 'target': 'drops in curls/pushups'},
        ],
        'feedback': {
            'tone': 'neutral',
            'title': 'Move into a supported exercise shape',
            'details': [
                'This build recognizes squats, pushups, and bicep curls.',
                'Turn slightly side-on so the joint angles are easier to score.',
            ],
        },
    }


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
            'analysis': no_pose_analysis(),
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
                'analysis': no_pose_analysis(),
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
voice_manager = VoiceManager()
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
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                runtime.set_error('Could not open webcam from Python.')
                return

            try:
                analysis_samples: list[dict[str, Any]] = []
                next_publish_at = int(time.time() * 1000) + ANALYSIS_WINDOW_MS

                while runtime.running:
                    ok, frame = cap.read()
                    if not ok:
                        runtime.set_error('Failed to read a frame from the webcam.')
                        time.sleep(0.1)
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = int(time.time() * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    frame = draw_landmarks_on_frame(frame, result)

                    encoded, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if not encoded:
                        continue

                    joints = build_joint_payload(result)
                    analysis_samples.append({
                        'timestamp': timestamp_ms,
                        'analysis': analyze_pose(result),
                    })

                    published_analysis = runtime.snapshot().get('analysis', no_pose_analysis())
                    if timestamp_ms >= next_publish_at:
                        window_started_at = next_publish_at - ANALYSIS_WINDOW_MS
                        window_ended_at = next_publish_at
                        window_samples = [
                            sample
                            for sample in analysis_samples
                            if window_started_at <= sample['timestamp'] < window_ended_at
                        ]
                        published_analysis = aggregate_analysis_window(
                            window_samples,
                            window_started_at,
                            window_ended_at,
                        )
                        analysis_samples = [
                            sample for sample in analysis_samples if sample['timestamp'] >= window_ended_at
                        ]
                        next_publish_at += ANALYSIS_WINDOW_MS
                        voice_manager.request(build_spoken_feedback(published_analysis))

                    payload = {
                        'status': 'Camera live' if joints else 'Searching for pose...',
                        'error': '',
                        'updatedAt': timestamp_ms,
                        'frameWidth': int(frame.shape[1]),
                        'frameHeight': int(frame.shape[0]),
                        'landmarkCount': len(result.pose_landmarks[0]) if result and result.pose_landmarks else 0,
                        'joints': joints,
                        'analysis': published_analysis,
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
