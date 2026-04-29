export interface PoseLandmark {
  x: number
  y: number
  z: number
  visibility?: number
}

export const LANDMARK_INDEX = {
  nose: 0,
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

const SKELETON_CONNECTIONS: Array<[number, number]> = [
  [LANDMARK_INDEX.leftShoulder, LANDMARK_INDEX.rightShoulder],
  [LANDMARK_INDEX.leftShoulder, LANDMARK_INDEX.leftElbow],
  [LANDMARK_INDEX.leftElbow, LANDMARK_INDEX.leftWrist],
  [LANDMARK_INDEX.rightShoulder, LANDMARK_INDEX.rightElbow],
  [LANDMARK_INDEX.rightElbow, LANDMARK_INDEX.rightWrist],
  [LANDMARK_INDEX.leftShoulder, LANDMARK_INDEX.leftHip],
  [LANDMARK_INDEX.rightShoulder, LANDMARK_INDEX.rightHip],
  [LANDMARK_INDEX.leftHip, LANDMARK_INDEX.rightHip],
  [LANDMARK_INDEX.leftHip, LANDMARK_INDEX.leftKnee],
  [LANDMARK_INDEX.leftKnee, LANDMARK_INDEX.leftAnkle],
  [LANDMARK_INDEX.rightHip, LANDMARK_INDEX.rightKnee],
  [LANDMARK_INDEX.rightKnee, LANDMARK_INDEX.rightAnkle],
]

export function drawPose(
  ctx: CanvasRenderingContext2D,
  landmarks: PoseLandmark[] | undefined,
  width: number,
  height: number,
) {
  if (!landmarks) {
    return
  }

  ctx.lineWidth = Math.max(3, width * 0.004)

  for (const [start, end] of SKELETON_CONNECTIONS) {
    const from = landmarks[start]
    const to = landmarks[end]

    if (!isLandmarkVisible(from, 0.1) || !isLandmarkVisible(to, 0.1)) {
      continue
    }

    ctx.strokeStyle = 'rgba(92, 249, 170, 0.9)'
    ctx.beginPath()
    ctx.moveTo((1 - from.x) * width, from.y * height)
    ctx.lineTo((1 - to.x) * width, to.y * height)
    ctx.stroke()
  }

  for (const landmark of landmarks) {
    if (!isLandmarkVisible(landmark, 0.1)) {
      continue
    }

    const x = (1 - landmark.x) * width
    const y = landmark.y * height
    const radius = Math.max(5, width * 0.007)

    ctx.fillStyle =
      (landmark.visibility ?? 1) >= 0.5
        ? 'rgba(247, 250, 252, 0.98)'
        : 'rgba(255, 196, 107, 0.9)'

    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.fill()
  }
}

export function calculateAngle(pointA: PoseLandmark, pointB: PoseLandmark, pointC: PoseLandmark) {
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

export function calculateSegmentAngleFromVertical(lowerPoint: PoseLandmark, upperPoint: PoseLandmark) {
  const deltaX = upperPoint.x - lowerPoint.x
  const deltaY = upperPoint.y - lowerPoint.y

  return Math.round(Math.abs((Math.atan2(deltaX, -deltaY) * 180) / Math.PI))
}

export function getPrimarySide(landmarks: PoseLandmark[]) {
  const leftScore = averageVisibility(landmarks, TRACKED_LEFT_SIDE)
  const rightScore = averageVisibility(landmarks, TRACKED_RIGHT_SIDE)

  return leftScore >= rightScore ? 'left' : 'right'
}

export function isLandmarkVisible(
  landmark: PoseLandmark | undefined,
  minimumVisibility = 0.5,
) {
  return Boolean(
    landmark &&
      Number.isFinite(landmark.x) &&
      Number.isFinite(landmark.y) &&
      (landmark.visibility ?? 1) >= minimumVisibility,
  )
}

export function countVisibleLandmarks(landmarks: PoseLandmark[] | undefined, minimumVisibility = 0.15) {
  return landmarks?.filter((lm) => isLandmarkVisible(lm, minimumVisibility)).length ?? 0
}

function averageVisibility(landmarks: PoseLandmark[], indices: number[]) {
  const visibleLandmarks = indices
    .map((index) => landmarks[index])
    .filter((landmark): landmark is PoseLandmark => Boolean(landmark))

  if (visibleLandmarks.length === 0) {
    return 0
  }

  const total = visibleLandmarks.reduce((sum, lm) => sum + (lm.visibility ?? 1), 0)

  return total / visibleLandmarks.length
}
