import { useEffect, useRef, useState } from 'react'
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'

import './App.css'

import {
  analyzeExercise,
  createNoPoseAnalysis,
  type ExerciseAnalysis,
  type ExerciseType,
} from './lib/feedback'
import { drawPose, isLandmarkVisible } from './lib/pose'

const MEDIAPIPE_VERSION = '0.10.35'
const MODEL_ASSET_PATH =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
const WASM_ASSET_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`

type ModelStatus = 'loading' | 'ready' | 'error'
type CameraStatus = 'idle' | 'starting' | 'live' | 'error'

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const isLoopRunningRef = useRef(false)
  const lastVideoTimeRef = useRef(-1)
  const repBottomReachedRef = useRef(false)
  const selectedExerciseRef = useRef<ExerciseType>('squat')
  const spokenCueRef = useRef<{ text: string; at: number }>({ text: '', at: 0 })

  const [modelStatus, setModelStatus] = useState<ModelStatus>('loading')
  const [modelError, setModelError] = useState('')
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('idle')
  const [cameraError, setCameraError] = useState('')
  const [selectedExercise, setSelectedExercise] = useState<ExerciseType>('squat')
  const [audioEnabled, setAudioEnabled] = useState(true)
  const [repCount, setRepCount] = useState(0)
  const [analysis, setAnalysis] = useState<ExerciseAnalysis>(createNoPoseAnalysis('squat'))
  const [landmarkCount, setLandmarkCount] = useState(0)
  const [poseVisible, setPoseVisible] = useState(false)

  useEffect(() => {
    selectedExerciseRef.current = selectedExercise
  }, [selectedExercise])

  useEffect(() => {
    let disposed = false

    async function loadModel() {
      try {
        setModelStatus('loading')
        const vision = await FilesetResolver.forVisionTasks(WASM_ASSET_PATH)
        const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_ASSET_PATH,
          },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.55,
          minPosePresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
        })

        if (disposed) {
          poseLandmarker.close()
          return
        }

        poseLandmarkerRef.current = poseLandmarker
        setModelStatus('ready')
      } catch (error) {
        console.error(error)

        if (!disposed) {
          setModelStatus('error')
          setModelError('The pose model could not load. Check your connection and refresh.')
        }
      }
    }

    loadModel()

    return () => {
      disposed = true
      stopCamera()
      poseLandmarkerRef.current?.close()
      poseLandmarkerRef.current = null
      window.speechSynthesis?.cancel()
    }
  }, [])

  useEffect(() => {
    if (!audioEnabled) {
      window.speechSynthesis?.cancel()
    }
  }, [audioEnabled])

  useEffect(() => {
    if (!audioEnabled || cameraStatus !== 'live' || !analysis.feedback.spokenCue) {
      return
    }

    const now = Date.now()
    const lastCue = spokenCueRef.current

    if (lastCue.text === analysis.feedback.spokenCue && now - lastCue.at < 4500) {
      return
    }

    spokenCueRef.current = {
      text: analysis.feedback.spokenCue,
      at: now,
    }

    const utterance = new SpeechSynthesisUtterance(analysis.feedback.spokenCue)
    utterance.rate = 1
    utterance.pitch = 0.95
    window.speechSynthesis?.cancel()
    window.speechSynthesis?.speak(utterance)
  }, [analysis.feedback.spokenCue, audioEnabled, cameraStatus])

  async function startCamera() {
    try {
      setCameraStatus('starting')
      setCameraError('')

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

      if (!video) {
        throw new Error('Video element unavailable')
      }

      video.srcObject = stream
      video.autoplay = true
      await video.play()

      lastVideoTimeRef.current = -1
      setCameraStatus('live')
      startProcessingLoop()
    } catch (error) {
      console.error(error)
      stopCamera()
      setCameraStatus('error')
      setCameraError('Camera access failed. Allow camera access and try again.')
    }
  }

  function stopCamera() {
    isLoopRunningRef.current = false

    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    lastVideoTimeRef.current = -1

    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.srcObject = null
    }

    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')

    if (canvas && context) {
      context.clearRect(0, 0, canvas.width, canvas.height)
    }

    setCameraStatus('idle')
    setAnalysis(createNoPoseAnalysis(selectedExerciseRef.current))
    setLandmarkCount(0)
    setPoseVisible(false)
    window.speechSynthesis?.cancel()
  }

  function startProcessingLoop() {
    if (isLoopRunningRef.current) {
      return
    }

    isLoopRunningRef.current = true

    const processFrame = () => {
      if (!isLoopRunningRef.current) {
        return
      }

      const video = videoRef.current
      const canvas = canvasRef.current
      const poseLandmarker = poseLandmarkerRef.current

      if (!video || !canvas || !poseLandmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        animationFrameRef.current = requestAnimationFrame(processFrame)
        return
      }

      if (video.currentTime === lastVideoTimeRef.current) {
        animationFrameRef.current = requestAnimationFrame(processFrame)
        return
      }

      lastVideoTimeRef.current = video.currentTime

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }

      const context = canvas.getContext('2d')

      if (!context) {
        animationFrameRef.current = requestAnimationFrame(processFrame)
        return
      }

      const result = poseLandmarker.detectForVideo(video, video.currentTime * 1000)
      const landmarks = result.landmarks[0]
      const visibleCount = landmarks?.filter((landmark) => isLandmarkVisible(landmark, 0.15)).length ?? 0

      drawPose(context, landmarks, canvas.width, canvas.height)

      const nextAnalysis = analyzeExercise(selectedExerciseRef.current, landmarks)
      handleRepCounting(nextAnalysis)
      setLandmarkCount(visibleCount)
      setPoseVisible(visibleCount >= 8)
      setAnalysis(nextAnalysis)

      animationFrameRef.current = requestAnimationFrame(processFrame)
    }

    animationFrameRef.current = requestAnimationFrame(processFrame)
  }

  function handleRepCounting(nextAnalysis: ExerciseAnalysis) {
    if (!nextAnalysis.poseDetected) {
      return
    }

    if (nextAnalysis.stage === 'bottom') {
      repBottomReachedRef.current = true
      return
    }

    if (nextAnalysis.stage === 'top' && repBottomReachedRef.current) {
      repBottomReachedRef.current = false
      setRepCount((current) => current + 1)
    }
  }

  function handleExerciseChange(exercise: ExerciseType) {
    selectedExerciseRef.current = exercise
    repBottomReachedRef.current = false
    setSelectedExercise(exercise)
    setRepCount(0)
    setLandmarkCount(0)
    setPoseVisible(false)
    setAnalysis(createNoPoseAnalysis(exercise))
  }

  const isCameraLive = cameraStatus === 'live'

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <p className="eyebrow">ConHacks 2026 MVP</p>
        <h1>Gym Bro</h1>
        <p className="hero-copy">
          Live exercise coaching with a webcam feed, pose landmarks, joint-angle checks,
          and spoken correction cues for squats and pushups.
        </p>
      </section>

      <section className="workspace">
        <div className="viewer-panel">
          <div className="toolbar">
            <div className="button-row">
              <button
                type="button"
                className="primary-button"
                onClick={isCameraLive ? stopCamera : startCamera}
                disabled={cameraStatus === 'starting' || modelStatus === 'loading'}
              >
                {isCameraLive ? 'Stop camera' : cameraStatus === 'starting' ? 'Starting...' : 'Start camera'}
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => setAudioEnabled((enabled) => !enabled)}
              >
                {audioEnabled ? 'Voice cues on' : 'Voice cues off'}
              </button>
            </div>

            <div className="exercise-picker" role="tablist" aria-label="Exercise selection">
              {(['squat', 'pushup'] as ExerciseType[]).map((exercise) => (
                <button
                  key={exercise}
                  type="button"
                  className={exercise === selectedExercise ? 'chip active' : 'chip'}
                  onClick={() => handleExerciseChange(exercise)}
                >
                  {exercise}
                </button>
              ))}
            </div>
          </div>

          <div className="camera-stage">
            <video ref={videoRef} className="camera-layer" playsInline muted />
            <canvas ref={canvasRef} className="overlay-layer" />

            {!isCameraLive && (
              <div className="stage-overlay">
                <p>Camera preview will appear here.</p>
                <span>Use a side view and keep your full body visible for the best tracking.</span>
              </div>
            )}
          </div>
        </div>

        <aside className="coach-panel">
          <div className="status-grid">
            <article className="status-card">
              <span className="label">Model</span>
              <strong>{modelStatus}</strong>
              <p>{modelError || 'MediaPipe Pose Landmarker is used for real-time tracking.'}</p>
            </article>

            <article className="status-card">
              <span className="label">Camera</span>
              <strong>{cameraStatus}</strong>
              <p>{cameraError || 'Front camera feed is processed directly in the browser.'}</p>
            </article>

            <article className="status-card compact">
              <span className="label">Exercise</span>
              <strong>{selectedExercise}</strong>
            </article>

            <article className="status-card compact">
              <span className="label">Reps</span>
              <strong>{repCount}</strong>
            </article>

            <article className="status-card compact">
              <span className="label">Phase</span>
              <strong>{analysis.stage}</strong>
            </article>

            <article className="status-card compact">
              <span className="label">Pose</span>
              <strong>{poseVisible ? 'detected' : 'searching'}</strong>
            </article>

            <article className="status-card compact">
              <span className="label">Landmarks</span>
              <strong>{landmarkCount}</strong>
            </article>
          </div>

          <article className={`feedback-card tone-${analysis.feedback.tone}`}>
            <span className="label">Live feedback</span>
            <h2>{analysis.feedback.title}</h2>
            <ul>
              {analysis.feedback.details.map((detail) => (
                <li key={detail}>{detail}</li>
              ))}
            </ul>
          </article>

          <article className="metrics-card">
            <span className="label">Tracked angles</span>
            <div className="metrics-list">
              {analysis.metrics.length > 0 ? (
                analysis.metrics.map((metric) => (
                  <div key={metric.label} className="metric-row">
                    <div>
                      <strong>{metric.label}</strong>
                      <span>{metric.target}</span>
                    </div>
                    <strong>{metric.value} deg</strong>
                  </div>
                ))
              ) : (
                <p className="empty-copy">Angles will appear once the model can see a clear pose.</p>
              )}
            </div>
          </article>

          <article className="notes-card">
            <span className="label">Current MVP scope</span>
            <ul>
              <li>Camera feed</li>
              <li>Pose detection model</li>
              <li>Skeleton and keypoint overlay</li>
              <li>Joint-angle and phase calculations</li>
              <li>Rule-based feedback with optional spoken cues</li>
            </ul>
          </article>
        </aside>
      </section>
    </main>
  )
}

export default App
