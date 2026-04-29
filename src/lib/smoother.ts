import type { PoseLandmark } from './pose'

/**
 * Causal 7-frame moving average for real-time display.
 * x̄_i = (1/n) * Σ x_j,  j = i-(n-1)..i
 */
export function smoothLandmarksRealtime(
  history: PoseLandmark[][],
  windowSize = 7,
): PoseLandmark[] | undefined {
  if (history.length === 0) return undefined
  const window = history.length <= windowSize ? history : history.slice(history.length - windowSize)
  return averageWindow(window)
}

/**
 * Symmetric centered moving average over all recorded frames for export.
 * x̄_i = (1/n) * Σ x_j,  j = i-h..i+h,  h = floor(n/2)
 */
export function smoothAllFrames(
  frames: PoseLandmark[][],
  windowSize = 7,
): PoseLandmark[][] {
  const h = Math.floor(windowSize / 2)
  return frames.map((_, i) => {
    const start = Math.max(0, i - h)
    const end = Math.min(frames.length - 1, i + h)
    return averageWindow(frames.slice(start, end + 1))!
  })
}

function averageWindow(window: PoseLandmark[][]): PoseLandmark[] {
  const n = window.length
  const landmarkCount = window[0].length
  return Array.from({ length: landmarkCount }, (_, li) => {
    let x = 0, y = 0, z = 0
    for (const frame of window) {
      const lm = frame[li]
      x += lm?.x ?? 0
      y += lm?.y ?? 0
      z += lm?.z ?? 0
    }
    const last = window[window.length - 1][li]
    return { x: x / n, y: y / n, z: z / n, visibility: last?.visibility }
  })
}
