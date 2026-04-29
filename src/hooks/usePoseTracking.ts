import { useRef, useState, useEffect, useCallback } from 'react'
import { FilesetResolver, PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'

// ─── Public types ────────────────────────────────────────────────────────────

export type ModelStatus  = 'loading' | 'ready' | 'error'
export type CameraStatus = 'idle' | 'starting' | 'live' | 'error'

/** A single named joint with normalised coords and a screen-pixel position. */
export interface Joint {
  /** 0–1, already mirrored to match the selfie display */
  x: number
  y: number
  /** Relative depth from the model */
  z: number
  /** Confidence that this joint is visible */
  visibility: number
  /** Canvas pixel position (updates with canvas size) */
  screenX: number
  screenY: number
}

/** All tracked joints, updated every frame as the user moves. */
export interface PoseData {
  nose:          Joint
  leftShoulder:  Joint
  rightShoulder: Joint
  leftElbow:     Joint
  rightElbow:    Joint
  leftWrist:     Joint
  rightWrist:    Joint
  leftHip:       Joint
  rightHip:      Joint
  leftKnee:      Joint
  rightKnee:     Joint
  leftAnkle:     Joint
  rightAnkle:    Joint
  /** Raw 33-landmark array straight from MediaPipe */
  raw: NormalizedLandmark[]
  /** performance.now() timestamp of this frame */
  capturedAt: number
}

// ─── MediaPipe indices ───────────────────────────────────────────────────────

const IDX = {
  nose:          0,
  leftShoulder:  11,
  rightShoulder: 12,
  leftElbow:     13,
  rightElbow:    14,
  leftWrist:     15,
  rightWrist:    16,
  leftHip:       23,
  rightHip:      24,
  leftKnee:      25,
  rightKnee:     26,
  leftAnkle:     27,
  rightAnkle:    28,
} as const

// ─── Skeleton connections ────────────────────────────────────────────────────

const CONNECTIONS: Array<[keyof typeof IDX, keyof typeof IDX]> = [
  ['leftShoulder',  'rightShoulder'],
  ['leftShoulder',  'leftElbow'],
  ['leftElbow',     'leftWrist'],
  ['rightShoulder', 'rightElbow'],
  ['rightElbow',    'rightWrist'],
  ['leftShoulder',  'leftHip'],
  ['rightShoulder', 'rightHip'],
  ['leftHip',       'rightHip'],
  ['leftHip',       'leftKnee'],
  ['leftKnee',      'leftAnkle'],
  ['rightHip',      'rightKnee'],
  ['rightKnee',     'rightAnkle'],
  ['nose',          'leftShoulder'],
  ['nose',          'rightShoulder'],
]

// ─── MediaPipe setup ─────────────────────────────────────────────────────────

const WASM_PATH  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'

async function createLandmarker(delegate: 'GPU' | 'CPU'): Promise<PoseLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH)
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  })
}

// ─── Drawing ─────────────────────────────────────────────────────────────────

function isVisible(lm: NormalizedLandmark): boolean {
  return (
    Number.isFinite(lm.x) &&
    Number.isFinite(lm.y) &&
    (lm.visibility ?? 1) >= 0.1
  )
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  raw: NormalizedLandmark[],
  w: number,
  h: number,
) {
  const px = (lm: NormalizedLandmark) => ({ x: (1 - lm.x) * w, y: lm.y * h })

  // Lines between connected joints
  ctx.lineWidth  = Math.max(2, w * 0.003)
  ctx.strokeStyle = '#5cf9aa'
  ctx.lineCap    = 'round'

  for (const [nameA, nameB] of CONNECTIONS) {
    const a = raw[IDX[nameA]]
    const b = raw[IDX[nameB]]
    if (!a || !b || !isVisible(a) || !isVisible(b)) continue
    const pa = px(a)
    const pb = px(b)
    ctx.beginPath()
    ctx.moveTo(pa.x, pa.y)
    ctx.lineTo(pb.x, pb.y)
    ctx.stroke()
  }

  // Dots at each joint
  const dotRadius = Math.max(4, w * 0.006)

  for (const raw_lm of raw) {
    if (!isVisible(raw_lm)) continue
    const { x, y } = px(raw_lm)
    const highConf = (raw_lm.visibility ?? 1) >= 0.5

    ctx.beginPath()
    ctx.arc(x, y, dotRadius, 0, Math.PI * 2)
    ctx.fillStyle = highConf ? 'rgba(247, 250, 252, 0.95)' : 'rgba(255, 196, 107, 0.9)'
    ctx.fill()
  }
}

// ─── Build PoseData ───────────────────────────────────────────────────────────

function buildPoseData(raw: NormalizedLandmark[], w: number, h: number): PoseData {
  const joint = (name: keyof typeof IDX): Joint => {
    const lm = raw[IDX[name]]
    const mx = lm ? (1 - lm.x) : 0     // mirror x to match display
    const my = lm?.y ?? 0
    return {
      x:         mx,
      y:         my,
      z:         lm?.z ?? 0,
      visibility: lm?.visibility ?? 0,
      screenX:   mx * w,
      screenY:   my * h,
    }
  }

  return {
    nose:          joint('nose'),
    leftShoulder:  joint('leftShoulder'),
    rightShoulder: joint('rightShoulder'),
    leftElbow:     joint('leftElbow'),
    rightElbow:    joint('rightElbow'),
    leftWrist:     joint('leftWrist'),
    rightWrist:    joint('rightWrist'),
    leftHip:       joint('leftHip'),
    rightHip:      joint('rightHip'),
    leftKnee:      joint('leftKnee'),
    rightKnee:     joint('rightKnee'),
    leftAnkle:     joint('leftAnkle'),
    rightAnkle:    joint('rightAnkle'),
    raw,
    capturedAt: performance.now(),
  }
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function usePoseTracking() {
  const videoRef  = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const landmarkerRef     = useRef<PoseLandmarker | null>(null)
  const streamRef         = useRef<MediaStream | null>(null)
  const rafRef            = useRef<number | null>(null)
  const loopActiveRef     = useRef(false)
  const lastVideoTimeRef  = useRef(-1)
  const frameIndexRef     = useRef(0)

  const [modelStatus,  setModelStatus]  = useState<ModelStatus>('loading')
  const [modelError,   setModelError]   = useState('')
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('idle')
  const [cameraError,  setCameraError]  = useState('')
  const [delegate,     setDelegate]     = useState<'GPU' | 'CPU' | null>(null)

  /**
   * Live pose data — updates every frame as the user moves.
   * null when no pose is detected or camera is off.
   */
  const [poseData, setPoseData] = useState<PoseData | null>(null)

  // Load model on mount
  useEffect(() => {
    let disposed = false

    async function load() {
      setModelStatus('loading')
      try {
        let lm: PoseLandmarker
        let mode: 'GPU' | 'CPU'
        try   { lm = await createLandmarker('GPU'); mode = 'GPU' }
        catch { lm = await createLandmarker('CPU'); mode = 'CPU' }

        if (disposed) { lm.close(); return }
        landmarkerRef.current = lm
        setDelegate(mode)
        setModelStatus('ready')
      } catch (err) {
        console.error(err)
        if (!disposed) {
          setModelStatus('error')
          setModelError('Model failed to load — check your connection and refresh.')
        }
      }
    }

    load()

    return () => {
      disposed = true
      stopCameraInternal()
      landmarkerRef.current?.close()
      landmarkerRef.current = null
    }
  }, [])

  // ── Processing loop ─────────────────────────────────────────────────────

  const startLoop = useCallback(() => {
    if (loopActiveRef.current) return
    loopActiveRef.current = true

    const tick = () => {
      if (!loopActiveRef.current) return

      // Always reschedule first so errors can't kill the loop
      rafRef.current = requestAnimationFrame(tick)

      const video    = videoRef.current
      const canvas   = canvasRef.current
      const lmarker  = landmarkerRef.current

      if (!video || !canvas || !lmarker) return

      // Guard: dimensions must be non-zero before detection is valid
      if (video.videoWidth === 0 || video.videoHeight === 0) return

      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return
      if (video.currentTime === lastVideoTimeRef.current) return
      lastVideoTimeRef.current = video.currentTime

      const w = video.videoWidth
      const h = video.videoHeight
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h }

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // Draw mirrored camera feed
      ctx.save()
      ctx.translate(w, 0)
      ctx.scale(-1, 1)
      ctx.drawImage(video, 0, 0, w, h)
      ctx.restore()

      // Detect pose
      let raw: NormalizedLandmark[] | undefined
      try {
        raw = lmarker.detectForVideo(video, performance.now()).landmarks[0]
      } catch { return }

      // Draw skeleton lines + dots over the video
      if (raw) drawSkeleton(ctx, raw, w, h)

      // Update poseData state — throttled to ~15 fps to avoid excess re-renders
      frameIndexRef.current++
      if (raw && frameIndexRef.current % 4 === 0) {
        setPoseData(buildPoseData(raw, w, h))
      } else if (!raw && frameIndexRef.current % 4 === 0) {
        setPoseData(null)
      }
    }

    rafRef.current = requestAnimationFrame(tick)
  }, [])

  // ── Camera controls ─────────────────────────────────────────────────────

  function stopCameraInternal() {
    loopActiveRef.current = false
    if (rafRef.current !== null) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    lastVideoTimeRef.current = -1
    if (videoRef.current) { videoRef.current.pause(); videoRef.current.srcObject = null }
    const canvas = canvasRef.current
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
  }

  const startCamera = useCallback(async () => {
    try {
      setCameraStatus('starting')
      setCameraError('')

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      streamRef.current = stream

      const video = videoRef.current
      if (!video) throw new Error('Video element not mounted')

      video.srcObject = stream
      video.muted     = true
      video.playsInline = true

      // Wait for the browser to report actual video dimensions before
      // starting the loop. play() resolving is not enough — videoWidth
      // can still be 0 until loadedmetadata fires.
      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve()
        video.onerror = () => reject(new Error('Video metadata failed to load'))
      })

      await video.play()

      lastVideoTimeRef.current = -1
      frameIndexRef.current    = 0
      setCameraStatus('live')
      startLoop()
    } catch (err) {
      console.error(err)
      stopCameraInternal()
      setCameraStatus('error')
      setCameraError('Camera access failed — allow camera permission and try again.')
    }
  }, [startLoop])

  const stopCamera = useCallback(() => {
    stopCameraInternal()
    setCameraStatus('idle')
    setPoseData(null)
  }, [])

  return {
    videoRef,
    canvasRef,
    modelStatus,
    modelError,
    cameraStatus,
    cameraError,
    delegate,
    poseData,
    startCamera,
    stopCamera,
  }
}
