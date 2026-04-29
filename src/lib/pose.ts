import { PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'

export const LANDMARK_INDEX = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
} as const

const TRACKED_LEFT_SIDE = [
  LANDMARK_INDEX.leftShoulder,
  LANDMARK_INDEX.leftHip,
  LANDMARK_INDEX.leftKnee,
  LANDMARK_INDEX.leftAnkle,
]

const TRACKED_RIGHT_SIDE = [
  LANDMARK_INDEX.rightShoulder,
  LANDMARK_INDEX.rightHip,
  LANDMARK_INDEX.rightKnee,
  LANDMARK_INDEX.rightAnkle,
]

export function drawPose(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[] | undefined,
  width: number,
  height: number,
) {
  ctx.clearRect(0, 0, width, height)

  if (!landmarks) {
    return
  }

  ctx.lineWidth = Math.max(2, width * 0.0035)
  ctx.strokeStyle = 'rgba(92, 249, 170, 0.82)'
  ctx.fillStyle = 'rgba(247, 250, 252, 0.95)'

  for (const connection of PoseLandmarker.POSE_CONNECTIONS) {
    const from = landmarks[connection.start]
    const to = landmarks[connection.end]

    if (!isLandmarkVisible(from, 0.2) || !isLandmarkVisible(to, 0.2)) {
      continue
    }

    ctx.beginPath()
    ctx.moveTo(from.x * width, from.y * height)
    ctx.lineTo(to.x * width, to.y * height)
    ctx.stroke()
  }

  for (const landmark of landmarks) {
    if (!isLandmarkVisible(landmark, 0.15)) {
      continue
    }

    const x = landmark.x * width
    const y = landmark.y * height
    const radius = Math.max(4, width * 0.006)

    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.fill()
  }
}

export function calculateAngle(
  pointA: NormalizedLandmark,
  pointB: NormalizedLandmark,
  pointC: NormalizedLandmark,
) {
  const vectorBA = {
    x: pointA.x - pointB.x,
    y: pointA.y - pointB.y,
  }

  const vectorBC = {
    x: pointC.x - pointB.x,
    y: pointC.y - pointB.y,
  }

  const dotProduct = vectorBA.x * vectorBC.x + vectorBA.y * vectorBC.y
  const magnitudeBA = Math.hypot(vectorBA.x, vectorBA.y)
  const magnitudeBC = Math.hypot(vectorBC.x, vectorBC.y)

  if (magnitudeBA === 0 || magnitudeBC === 0) {
    return 0
  }

  const cosine = dotProduct / (magnitudeBA * magnitudeBC)
  const safeCosine = Math.min(1, Math.max(-1, cosine))

  return Math.round((Math.acos(safeCosine) * 180) / Math.PI)
}

export function calculateSegmentAngleFromVertical(
  lowerPoint: NormalizedLandmark,
  upperPoint: NormalizedLandmark,
) {
  const deltaX = upperPoint.x - lowerPoint.x
  const deltaY = upperPoint.y - lowerPoint.y

  return Math.round(Math.abs((Math.atan2(deltaX, -deltaY) * 180) / Math.PI))
}

export function getPrimarySide(landmarks: NormalizedLandmark[]) {
  const leftScore = averageVisibility(landmarks, TRACKED_LEFT_SIDE)
  const rightScore = averageVisibility(landmarks, TRACKED_RIGHT_SIDE)

  return leftScore >= rightScore ? 'left' : 'right'
}

export function isLandmarkVisible(
  landmark: NormalizedLandmark | undefined,
  minimumVisibility = 0.5,
) {
  return Boolean(
    landmark &&
      Number.isFinite(landmark.x) &&
      Number.isFinite(landmark.y) &&
      landmark.visibility >= minimumVisibility,
  )
}

function averageVisibility(landmarks: NormalizedLandmark[], indices: number[]) {
  const visibleLandmarks = indices
    .map((index) => landmarks[index])
    .filter((landmark): landmark is NormalizedLandmark => Boolean(landmark))

  if (visibleLandmarks.length === 0) {
    return 0
  }

  const total = visibleLandmarks.reduce((sum, landmark) => sum + landmark.visibility, 0)

  return total / visibleLandmarks.length
}
