import { LANDMARK_INDEX, isLandmarkVisible, type PoseLandmark } from './pose'

interface Point {
  x: number
  y: number
  ok: boolean
}

function toPoint(landmarks: PoseLandmark[], index: number, w: number, h: number): Point {
  const lm = landmarks[index]
  if (!lm) return { x: 0, y: 0, ok: false }
  return {
    x: (1 - lm.x) * w,
    y: lm.y * h,
    ok: isLandmarkVisible(lm, 0.1),
  }
}

function mid(a: Point, b: Point): Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, ok: a.ok && b.ok }
}

function segment(
  ctx: CanvasRenderingContext2D,
  a: Point,
  b: Point,
  lineWidth: number,
  color: string,
) {
  if (!a.ok || !b.ok) return
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.beginPath()
  ctx.moveTo(a.x, a.y)
  ctx.lineTo(b.x, b.y)
  ctx.stroke()
  ctx.restore()
}

function joint(ctx: CanvasRenderingContext2D, p: Point, radius: number, color: string) {
  if (!p.ok) return
  ctx.save()
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

export function drawPuppet(
  ctx: CanvasRenderingContext2D,
  landmarks: PoseLandmark[],
  width: number,
  height: number,
) {
  // Dark background for the Stage
  ctx.fillStyle = '#07111f'
  ctx.fillRect(0, 0, width, height)

  const s = width / 640

  const L = LANDMARK_INDEX
  const nose      = toPoint(landmarks, L.nose,          width, height)
  const lShoulder = toPoint(landmarks, L.leftShoulder,  width, height)
  const rShoulder = toPoint(landmarks, L.rightShoulder, width, height)
  const lElbow    = toPoint(landmarks, L.leftElbow,     width, height)
  const rElbow    = toPoint(landmarks, L.rightElbow,    width, height)
  const lWrist    = toPoint(landmarks, L.leftWrist,     width, height)
  const rWrist    = toPoint(landmarks, L.rightWrist,    width, height)
  const lHip      = toPoint(landmarks, L.leftHip,       width, height)
  const rHip      = toPoint(landmarks, L.rightHip,      width, height)
  const lKnee     = toPoint(landmarks, L.leftKnee,      width, height)
  const rKnee     = toPoint(landmarks, L.rightKnee,     width, height)
  const lAnkle    = toPoint(landmarks, L.leftAnkle,     width, height)
  const rAnkle    = toPoint(landmarks, L.rightAnkle,    width, height)

  const shoulderMid = mid(lShoulder, rShoulder)
  const hipMid      = mid(lHip, rHip)

  // Torso quad (filled polygon between shoulders and hips)
  if (lShoulder.ok && rShoulder.ok && lHip.ok && rHip.ok) {
    ctx.save()
    ctx.fillStyle   = 'rgba(80, 140, 255, 0.18)'
    ctx.strokeStyle = 'rgba(80, 140, 255, 0.55)'
    ctx.lineWidth   = 1.5 * s
    ctx.lineJoin    = 'round'
    ctx.beginPath()
    ctx.moveTo(lShoulder.x, lShoulder.y)
    ctx.lineTo(rShoulder.x, rShoulder.y)
    ctx.lineTo(rHip.x, rHip.y)
    ctx.lineTo(lHip.x, lHip.y)
    ctx.closePath()
    ctx.fill()
    ctx.stroke()
    ctx.restore()
  }

  const GREEN  = '#5cf9aa'
  const PURPLE = '#b06fff'
  const AMBER  = '#ffb84d'

  // Spine
  segment(ctx, shoulderMid, hipMid, 6 * s, PURPLE)

  // Neck → nose
  if (nose.ok) segment(ctx, shoulderMid, nose, 8 * s, GREEN)

  // Upper arms (thicker) → forearms (thinner)
  segment(ctx, lShoulder, lElbow, 13 * s, GREEN)
  segment(ctx, lElbow,    lWrist,  9 * s, GREEN)
  segment(ctx, rShoulder, rElbow, 13 * s, GREEN)
  segment(ctx, rElbow,    rWrist,  9 * s, GREEN)

  // Thighs (thicker) → shins (thinner)
  segment(ctx, lHip,  lKnee,  13 * s, GREEN)
  segment(ctx, lKnee, lAnkle,  9 * s, GREEN)
  segment(ctx, rHip,  rKnee,  13 * s, GREEN)
  segment(ctx, rKnee, rAnkle,  9 * s, GREEN)

  // Hinge joints over limb endpoints
  const hinges = [
    lShoulder, rShoulder,
    lElbow, rElbow,
    lWrist, rWrist,
    lHip, rHip,
    lKnee, rKnee,
    lAnkle, rAnkle,
  ]
  for (const p of hinges) joint(ctx, p, 7 * s, AMBER)

  // Head
  if (nose.ok) {
    ctx.save()
    ctx.fillStyle   = '#f5c88a'
    ctx.strokeStyle = GREEN
    ctx.lineWidth   = 2 * s
    ctx.beginPath()
    ctx.arc(nose.x, nose.y, 16 * s, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
    ctx.restore()
  }
}
