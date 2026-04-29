import { useRef, useState, useEffect, useCallback } from 'react'
import '@mediapipe/pose'
import type { Pose as MediaPipePose, NormalizedLandmark, Results } from '@mediapipe/pose'

const MEDIAPIPE_POSE_VERSION = '0.5.1675469404'

declare global {
  interface Window {
    Pose: typeof MediaPipePose
  }
}

export type ModelStatus = 'loading' | 'ready' | 'error'
export type CameraStatus = 'idle' | 'starting' | 'live' | 'error'

export interface Joint {
  x: number
  y: number
  z: number
  visibility: number
  screenX: number
  screenY: number
}

export interface PoseData {
  nose: Joint
  leftShoulder: Joint
  rightShoulder: Joint
  leftElbow: Joint
  rightElbow: Joint
  leftWrist: Joint
  rightWrist: Joint
  leftHip: Joint
  rightHip: Joint
  leftKnee: Joint
  rightKnee: Joint
  leftAnkle: Joint
  rightAnkle: Joint
  raw: NormalizedLandmark[]
  capturedAt: number
}

const IDX = {
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

const FULL_POSE_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12],
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32],
  [27, 31], [28, 32],
]

function isVisible(lm: NormalizedLandmark | undefined): boolean {
  return Boolean(
    lm &&
    Number.isFinite(lm.x) &&
    Number.isFinite(lm.y) &&
    (lm.visibility ?? 1) >= 0.1,
  )
}

function drawFrame(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  raw: NormalizedLandmark[] | undefined,
  w: number,
  h: number,
) {
  ctx.clearRect(0, 0, w, h)

  ctx.save()
  ctx.translate(w, 0)
  ctx.scale(-1, 1)
  ctx.drawImage(video, 0, 0, w, h)
  ctx.restore()

  if (!raw) return

  const px = (lm: NormalizedLandmark) => ({ x: (1 - lm.x) * w, y: lm.y * h })

  ctx.lineWidth = Math.max(2, w * 0.003)
  ctx.strokeStyle = '#5cf9aa'
  ctx.lineCap = 'round'

  for (const [startIndex, endIndex] of FULL_POSE_CONNECTIONS) {
    const a = raw[startIndex]
    const b = raw[endIndex]
    if (!isVisible(a) || !isVisible(b)) continue

    const pa = px(a)
    const pb = px(b)
    ctx.beginPath()
    ctx.moveTo(pa.x, pa.y)
    ctx.lineTo(pb.x, pb.y)
    ctx.stroke()
  }

  const dotRadius = Math.max(4, w * 0.006)
  for (const lm of raw) {
    if (!isVisible(lm)) continue

    const { x, y } = px(lm)
    const highConf = (lm.visibility ?? 1) >= 0.5
    ctx.beginPath()
    ctx.arc(x, y, dotRadius, 0, Math.PI * 2)
    ctx.fillStyle = highConf ? 'rgba(247, 250, 252, 0.95)' : 'rgba(255, 196, 107, 0.9)'
    ctx.fill()
  }
}

function buildPoseData(raw: NormalizedLandmark[], w: number, h: number): PoseData {
  const joint = (name: keyof typeof IDX): Joint => {
    const lm = raw[IDX[name]]
    const x = lm ? (1 - lm.x) : 0
    const y = lm?.y ?? 0

    return {
      x,
      y,
      z: lm?.z ?? 0,
      visibility: lm?.visibility ?? 1,
      screenX: x * w,
      screenY: y * h,
    }
  }

  return {
    nose: joint('nose'),
    leftShoulder: joint('leftShoulder'),
    rightShoulder: joint('rightShoulder'),
    leftElbow: joint('leftElbow'),
    rightElbow: joint('rightElbow'),
    leftWrist: joint('leftWrist'),
    rightWrist: joint('rightWrist'),
    leftHip: joint('leftHip'),
    rightHip: joint('rightHip'),
    leftKnee: joint('leftKnee'),
    rightKnee: joint('rightKnee'),
    leftAnkle: joint('leftAnkle'),
    rightAnkle: joint('rightAnkle'),
    raw,
    capturedAt: performance.now(),
  }
}

function getErrorMessage(err: unknown): string {
  if (err instanceof Error && err.message) return err.message
  if (typeof err === 'string') return err
  return 'Unknown model load error'
}

export function usePoseTracking() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const poseRef = useRef<MediaPipePose | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const rafRef = useRef<number | null>(null)
  const loopActiveRef = useRef(false)
  const inferenceBusyRef = useRef(false)
  const lastVideoTimeRef = useRef(-1)
  const frameIndexRef = useRef(0)

  const [modelStatus, setModelStatus] = useState<ModelStatus>('loading')
  const [modelError, setModelError] = useState('')
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('idle')
  const [cameraError, setCameraError] = useState('')
  const [delegate] = useState<'GPU' | 'CPU' | null>(null)
  const [poseData, setPoseData] = useState<PoseData | null>(null)

  useEffect(() => {
    let disposed = false

    async function load() {
      setModelStatus('loading')
      setModelError('')

      try {
        if (typeof window.Pose !== 'function') {
          throw new Error('MediaPipe Pose script did not register correctly')
        }

        const pose = new window.Pose({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${MEDIAPIPE_POSE_VERSION}/${file}`,
        })

        pose.setOptions({
          modelComplexity: 2,
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: false,
          minDetectionConfidence: 0.35,
          minTrackingConfidence: 0.35,
          selfieMode: false,
        })

        pose.onResults((results: Results) => {
          const video = videoRef.current
          const canvas = canvasRef.current
          if (!video || !canvas) return

          const w = video.videoWidth
          const h = video.videoHeight
          if (w === 0 || h === 0) return

          if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w
            canvas.height = h
          }

          const ctx = canvas.getContext('2d')
          if (!ctx) return

          const raw = results.poseLandmarks?.length ? results.poseLandmarks : undefined
          drawFrame(ctx, video, raw, w, h)

          frameIndexRef.current += 1
          if (frameIndexRef.current % 2 !== 0) return

          if (raw) setPoseData(buildPoseData(raw, w, h))
          else setPoseData(null)
        })

        await pose.initialize()
        if (disposed) {
          await pose.close()
          return
        }

        poseRef.current = pose
        setModelStatus('ready')
      } catch (err) {
        console.error(err)
        if (!disposed) {
          setModelStatus('error')
          setModelError(`Pose model failed to load: ${getErrorMessage(err)}`)
        }
      }
    }

    load()

    return () => {
      disposed = true
      stopCameraInternal()
      void poseRef.current?.close()
      poseRef.current = null
    }
  }, [])

  const startLoop = useCallback(() => {
    if (loopActiveRef.current) return
    loopActiveRef.current = true

    const tick = async () => {
      if (!loopActiveRef.current) return
      rafRef.current = requestAnimationFrame(() => {
        void tick()
      })

      const video = videoRef.current
      const pose = poseRef.current
      if (!video || !pose || inferenceBusyRef.current) return
      if (video.videoWidth === 0 || video.videoHeight === 0) return
      if (video.readyState < HTMLMediaElement.HAVE_ENOUGH_DATA) return
      if (video.currentTime === lastVideoTimeRef.current) return

      lastVideoTimeRef.current = video.currentTime
      inferenceBusyRef.current = true

      try {
        await pose.send({ image: video })
      } catch (err) {
        console.error(err)
      } finally {
        inferenceBusyRef.current = false
      }
    }

    rafRef.current = requestAnimationFrame(() => {
      void tick()
    })
  }, [])

  function stopCameraInternal() {
    loopActiveRef.current = false
    inferenceBusyRef.current = false

    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }

    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    lastVideoTimeRef.current = -1
    frameIndexRef.current = 0

    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.srcObject = null
    }

    const canvas = canvasRef.current
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
  }

  const startCamera = useCallback(async () => {
    try {
      setCameraStatus('starting')
      setCameraError('')
      setPoseData(null)

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      })

      streamRef.current = stream

      const video = videoRef.current
      if (!video) throw new Error('Video element not mounted')

      video.srcObject = stream
      video.autoplay = true
      video.muted = true
      video.playsInline = true

      await new Promise<void>((resolve, reject) => {
        if (video.readyState >= HTMLMediaElement.HAVE_METADATA && video.videoWidth > 0) {
          resolve()
          return
        }

        const handleLoadedMetadata = () => {
          cleanup()
          resolve()
        }
        const handleError = () => {
          cleanup()
          reject(new Error('Video metadata failed to load'))
        }
        const cleanup = () => {
          video.removeEventListener('loadedmetadata', handleLoadedMetadata)
          video.removeEventListener('error', handleError)
        }

        video.addEventListener('loadedmetadata', handleLoadedMetadata, { once: true })
        video.addEventListener('error', handleError, { once: true })
      })

      await video.play()

      lastVideoTimeRef.current = -1
      frameIndexRef.current = 0
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
