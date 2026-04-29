import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

import {
  LANDMARK_INDEX,
  calculateAngle,
  calculateSegmentAngleFromVertical,
  getPrimarySide,
  isLandmarkVisible,
} from './pose'

export type ExerciseType = 'squat' | 'pushup'
export type ExerciseStage = 'top' | 'mid' | 'bottom' | 'unknown'
export type FeedbackTone = 'neutral' | 'good' | 'warn' | 'bad'

export interface Metric {
  label: string
  value: number
  target: string
}

export interface FeedbackCard {
  tone: FeedbackTone
  title: string
  details: string[]
  spokenCue?: string
}

export interface ExerciseAnalysis {
  exercise: ExerciseType
  stage: ExerciseStage
  poseDetected: boolean
  metrics: Metric[]
  feedback: FeedbackCard
}

export function createNoPoseAnalysis(exercise: ExerciseType): ExerciseAnalysis {
  return {
    exercise,
    stage: 'unknown',
    poseDetected: false,
    metrics: [],
    feedback: {
      tone: 'neutral',
      title: 'Waiting for a clear side view',
      details: [
        'Keep your full body in frame.',
        'Stand side-on to the camera so the joint angles are easier to read.',
      ],
    },
  }
}

export function analyzeExercise(
  exercise: ExerciseType,
  landmarks: NormalizedLandmark[] | undefined,
) {
  if (!landmarks) {
    return createNoPoseAnalysis(exercise)
  }

  if (exercise === 'pushup') {
    return analyzePushup(landmarks)
  }

  return analyzeSquat(landmarks)
}

function analyzeSquat(landmarks: NormalizedLandmark[]): ExerciseAnalysis {
  const side = getPrimarySide(landmarks)
  const shoulder = landmarks[
    side === 'left' ? LANDMARK_INDEX.leftShoulder : LANDMARK_INDEX.rightShoulder
  ]
  const hip = landmarks[side === 'left' ? LANDMARK_INDEX.leftHip : LANDMARK_INDEX.rightHip]
  const knee = landmarks[side === 'left' ? LANDMARK_INDEX.leftKnee : LANDMARK_INDEX.rightKnee]
  const ankle = landmarks[
    side === 'left' ? LANDMARK_INDEX.leftAnkle : LANDMARK_INDEX.rightAnkle
  ]

  if (
    !isLandmarkVisible(shoulder, 0.3) ||
    !isLandmarkVisible(hip, 0.3) ||
    !isLandmarkVisible(knee, 0.3) ||
    !isLandmarkVisible(ankle, 0.3)
  ) {
    return createNoPoseAnalysis('squat')
  }

  const kneeAngle = calculateAngle(hip, knee, ankle)
  const hipAngle = calculateAngle(shoulder, hip, knee)
  const torsoLean = calculateSegmentAngleFromVertical(hip, shoulder)
  const stage = getStage(kneeAngle, 105, 155)
  const details: string[] = []
  let tone: FeedbackTone = 'good'
  let title = 'Strong squat position'
  let spokenCue: string | undefined

  if (stage !== 'top' && kneeAngle > 118) {
    details.push('Go a little deeper so your knees bend more at the bottom.')
    spokenCue ??= 'Go a little deeper into the squat.'
    tone = 'warn'
    title = 'Needs more depth'
  }

  if (torsoLean > 36) {
    details.push('Keep your chest taller and brace your core to avoid folding forward.')
    spokenCue ??= 'Keep your chest up and your core tight.'
    tone = tone === 'warn' ? 'bad' : 'warn'
    title = 'Chest is dropping forward'
  }

  if (stage === 'top' && tone === 'good') {
    tone = 'neutral'
    title = 'Ready for the next squat'
    details.push('Sit your hips back, bend your knees, then drive back up to full height.')
  }

  if (stage === 'bottom' && tone === 'good') {
    details.push('Depth looks solid. Drive through your feet and stand tall.')
  }

  if (stage === 'mid' && tone === 'good') {
    details.push('The rep path looks controlled. Keep your knees and hips moving together.')
  }

  return {
    exercise: 'squat',
    stage,
    poseDetected: true,
    metrics: [
      { label: 'Knee angle', value: kneeAngle, target: '90-110 deg bottom' },
      { label: 'Hip angle', value: hipAngle, target: 'stay controlled' },
      { label: 'Torso lean', value: torsoLean, target: 'under 35 deg' },
    ],
    feedback: {
      tone,
      title,
      details,
      spokenCue,
    },
  }
}

function analyzePushup(landmarks: NormalizedLandmark[]): ExerciseAnalysis {
  const side = getPrimarySide(landmarks)
  const shoulder = landmarks[
    side === 'left' ? LANDMARK_INDEX.leftShoulder : LANDMARK_INDEX.rightShoulder
  ]
  const elbow = landmarks[side === 'left' ? LANDMARK_INDEX.leftElbow : LANDMARK_INDEX.rightElbow]
  const wrist = landmarks[side === 'left' ? LANDMARK_INDEX.leftWrist : LANDMARK_INDEX.rightWrist]
  const hip = landmarks[side === 'left' ? LANDMARK_INDEX.leftHip : LANDMARK_INDEX.rightHip]
  const ankle = landmarks[
    side === 'left' ? LANDMARK_INDEX.leftAnkle : LANDMARK_INDEX.rightAnkle
  ]

  if (
    !isLandmarkVisible(shoulder, 0.3) ||
    !isLandmarkVisible(elbow, 0.3) ||
    !isLandmarkVisible(wrist, 0.3) ||
    !isLandmarkVisible(hip, 0.3) ||
    !isLandmarkVisible(ankle, 0.3)
  ) {
    return createNoPoseAnalysis('pushup')
  }

  const elbowAngle = calculateAngle(shoulder, elbow, wrist)
  const bodyLineAngle = calculateAngle(shoulder, hip, ankle)
  const stage = getStage(elbowAngle, 92, 152)
  const details: string[] = []
  let tone: FeedbackTone = 'good'
  let title = 'Pushup line looks strong'
  let spokenCue: string | undefined

  if (stage !== 'top' && elbowAngle > 105) {
    details.push('Lower a bit more so your elbows bend further at the bottom.')
    spokenCue ??= 'Lower your chest a little more.'
    tone = 'warn'
    title = 'Lower for more depth'
  }

  if (bodyLineAngle < 155) {
    details.push('Keep your body in one straight line from shoulders through ankles.')
    spokenCue ??= 'Keep your core tight and your body straight.'
    tone = tone === 'warn' ? 'bad' : 'warn'
    title = 'Core line is breaking'
  }

  if (stage === 'top' && tone === 'good') {
    tone = 'neutral'
    title = 'Ready for the next pushup'
    details.push('Brace your core, bend your elbows, and lower under control.')
  }

  if (stage === 'bottom' && tone === 'good') {
    details.push('Nice depth. Press straight back up without losing your body line.')
  }

  if (stage === 'mid' && tone === 'good') {
    details.push('Good control. Keep elbows bending smoothly with the torso moving as one unit.')
  }

  return {
    exercise: 'pushup',
    stage,
    poseDetected: true,
    metrics: [
      { label: 'Elbow angle', value: elbowAngle, target: '80-95 deg bottom' },
      { label: 'Body line', value: bodyLineAngle, target: '155-180 deg' },
    ],
    feedback: {
      tone,
      title,
      details,
      spokenCue,
    },
  }
}

function getStage(angle: number, bottomThreshold: number, topThreshold: number): ExerciseStage {
  if (angle <= bottomThreshold) {
    return 'bottom'
  }

  if (angle >= topThreshold) {
    return 'top'
  }

  return 'mid'
}
