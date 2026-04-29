import { useEffect, useRef, useState } from 'react'
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'

import './App.css'

import {
  analyzeExercise,
  createNoPoseAnalysis,
  type ExerciseAnalysis,
  type ExerciseType,
} from './lib/feedback'
import { countVisibleLandmarks, drawPose, type PoseLandmark } from './lib/pose'
import { drawPuppet } from './lib/stage'
import { smoothLandmarksRealtime, smoothAllFrames } from './lib/smoother'
import { exportJSON, exportHTML, type RecordedFrame } from './lib/exporter'

const MEDIAPIPE_VERSION = '0.10.35'
const MODEL_ASSET_PATH =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
const WASM_ASSET_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`

type ModelStatus = 'loading' | 'ready' | 'error'
type CameraStatus = 'idle' | 'starting' | 'live' | 'error'

async function buildLandmarker(delegate: 'GPU' | 'CPU'): Promise<PoseLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(WASM_ASSET_PATH)
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_ASSET_PATH, delegate },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  })
}

function App() {
  // Canvas / video refs
  const videoRef      = useRef<HTMLVideoElement | null>(null)
  const canvasRef     = useRef<HTMLCanvasElement | null>(null)  // MoCap Studio
  const stageCanvasRef = useRef<HTMLCanvasElement | null>(null) // Stage puppet

  // Core refs
  const streamRef         = useRef<MediaStream | null>(null)
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const isLoopRunningRef  = useRef(false)
  const lastVideoTimeRef  = useRef(-1)

  // Exercise refs
  const repBottomReachedRef = useRef(false)
  const selectedExerciseRef = useRef<ExerciseType>('squat')
  const spokenCueRef        = useRef<{ text: string; at: number }>({ text: '', at: 0 })

  // Smoothing — rolling window of raw landmarks for causal filter
  const landmarkHistoryRef  = useRef<PoseLandmark[][]>([])
  const smoothingEnabledRef = useRef(true)

  // Recording
  const framesRef       = useRef<RecordedFrame[]>([])
  const isRecordingRef  = useRef(false)

  // Playback
  const isPlayingRef      = useRef(false)
  const playAnimFrameRef  = useRef<number | null>(null)

  // UI state
  const [modelStatus, setModelStatus]     = useState<ModelStatus>('loading')
  const [modelError, setModelError]       = useState('')
  const [delegateMode, setDelegateMode]   = useState<'GPU' | 'CPU' | null>(null)
  const [cameraStatus, setCameraStatus]   = useState<CameraStatus>('idle')
  const [cameraError, setCameraError]     = useState('')
  const [selectedExercise, setSelectedExercise] = useState<ExerciseType>('squat')
  const [audioEnabled, setAudioEnabled]   = useState(true)
  const [smoothingEnabled, setSmoothingEnabled] = useState(true)
  const [repCount, setRepCount]           = useState(0)
  const [analysis, setAnalysis]           = useState<ExerciseAnalysis>(createNoPoseAnalysis('squat'))
  const [landmarkCount, setLandmarkCount] = useState(0)
  const [poseVisible, setPoseVisible]     = useState(false)
  const [isRecording, setIsRecording]     = useState(false)
  const [isPlaying, setIsPlaying]         = useState(false)
  const [frameCount, setFrameCount]       = useState(0)
  const [hasRecording, setHasRecording]   = useState(false)

  useEffect(() => { selectedExerciseRef.current = selectedExercise }, [selectedExercise])
  useEffect(() => { smoothingEnabledRef.current = smoothingEnabled }, [smoothingEnabled])

  // Load model — try GPU delegation first, fall back to CPU
  useEffect(() => {
    let disposed = false

    async function loadModel() {
      setModelStatus('loading')
      try {
        let pose: PoseLandmarker
        let mode: 'GPU' | 'CPU'
        try {
          pose = await buildLandmarker('GPU')
          mode = 'GPU'
        } catch {
          pose = await buildLandmarker('CPU')
          mode = 'CPU'
        }
        if (disposed) { pose.close(); return }
        poseLandmarkerRef.current = pose
        setDelegateMode(mode)
        setModelStatus('ready')
      } catch (err) {
        console.error(err)
        if (!disposed) {
          setModelStatus('error')
          setModelError('Pose model could not load. Check connection and refresh.')
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
    if (!audioEnabled) window.speechSynthesis?.cancel()
  }, [audioEnabled])

  useEffect(() => {
    if (!audioEnabled || cameraStatus !== 'live' || !analysis.feedback.spokenCue) return
    const now = Date.now()
    const last = spokenCueRef.current
    if (last.text === analysis.feedback.spokenCue && now - last.at < 4500) return
    spokenCueRef.current = { text: analysis.feedback.spokenCue, at: now }
    const utt = new SpeechSynthesisUtterance(analysis.feedback.spokenCue)
    utt.rate = 1; utt.pitch = 0.95
    window.speechSynthesis?.cancel()
    window.speechSynthesis?.speak(utt)
  }, [analysis.feedback.spokenCue, audioEnabled, cameraStatus])

  async function startCamera() {
    try {
      setCameraStatus('starting')
      setCameraError('')
      stopPlayback()

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      streamRef.current = stream

      const video = videoRef.current
      if (!video) throw new Error('Video element unavailable')
      video.srcObject = stream
      video.autoplay = true
      await video.play()

      lastVideoTimeRef.current = -1
      setCameraStatus('live')
      startProcessingLoop()
    } catch (err) {
      console.error(err)
      stopCamera()
      setCameraStatus('error')
      setCameraError('Camera access failed. Allow camera access and try again.')
    }
  }

  function stopCamera() {
    stopRecording()
    isLoopRunningRef.current = false
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    lastVideoTimeRef.current = -1
    if (videoRef.current) { videoRef.current.pause(); videoRef.current.srcObject = null }
    const canvas = canvasRef.current
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
    setCameraStatus('idle')
    setAnalysis(createNoPoseAnalysis(selectedExerciseRef.current))
    setLandmarkCount(0)
    setPoseVisible(false)
    landmarkHistoryRef.current = []
    window.speechSynthesis?.cancel()
  }

  function startProcessingLoop() {
    if (isLoopRunningRef.current) return
    isLoopRunningRef.current = true

    const processFrame = () => {
      if (!isLoopRunningRef.current) return

      // Schedule next frame FIRST so errors below never kill the loop
      animationFrameRef.current = requestAnimationFrame(processFrame)

      const video     = videoRef.current
      const canvas    = canvasRef.current
      const stageCanvas = stageCanvasRef.current
      const landmarker  = poseLandmarkerRef.current

      if (
        !video || !canvas || !landmarker ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
      ) return

      if (video.currentTime === lastVideoTimeRef.current) return
      lastVideoTimeRef.current = video.currentTime

      const w = video.videoWidth
      const h = video.videoHeight
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h }

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // ── MoCap Studio: draw mirrored camera feed ──────────────────
      ctx.save()
      ctx.translate(w, 0)
      ctx.scale(-1, 1)
      ctx.drawImage(video, 0, 0, w, h)
      ctx.restore()

      // ── Pose detection ───────────────────────────────────────────
      let rawLandmarks: PoseLandmark[] | undefined
      try {
        rawLandmarks = landmarker.detectForVideo(video, performance.now()).landmarks[0] as PoseLandmark[] | undefined
      } catch { return }

      // ── Smoothing (causal 7-frame moving average) ─────────────────
      let landmarks = rawLandmarks
      if (rawLandmarks) {
        const hist = landmarkHistoryRef.current
        hist.push(rawLandmarks)
        if (hist.length > 7) hist.shift()
        if (smoothingEnabledRef.current) {
          landmarks = smoothLandmarksRealtime(hist, 7) ?? rawLandmarks
        }
      }

      // ── MoCap Studio: skeleton overlay ───────────────────────────
      drawPose(ctx, landmarks, w, h)

      // ── Record raw landmarks + timestamp ─────────────────────────
      if (isRecordingRef.current && rawLandmarks) {
        framesRef.current.push({ landmarks: rawLandmarks, timestamp: performance.now() })
        setFrameCount(framesRef.current.length)
      }

      // ── Stage: procedural puppet ──────────────────────────────────
      if (stageCanvas) {
        if (stageCanvas.width !== w || stageCanvas.height !== h) {
          stageCanvas.width = w; stageCanvas.height = h
        }
        const sc = stageCanvas.getContext('2d')
        if (sc) {
          if (landmarks) drawPuppet(sc, landmarks, stageCanvas.width, stageCanvas.height)
          else { sc.fillStyle = '#07111f'; sc.fillRect(0, 0, stageCanvas.width, stageCanvas.height) }
        }
      }

      // ── Form analysis + rep counting ──────────────────────────────
      const next = analyzeExercise(selectedExerciseRef.current, landmarks)
      handleRepCounting(next)
      const visible = countVisibleLandmarks(landmarks)
      setLandmarkCount(visible)
      setPoseVisible(visible >= 6)
      setAnalysis(next)
    }

    animationFrameRef.current = requestAnimationFrame(processFrame)
  }

  function handleRepCounting(next: ExerciseAnalysis) {
    if (!next.poseDetected) return
    if (next.stage === 'bottom') { repBottomReachedRef.current = true; return }
    if (next.stage === 'top' && repBottomReachedRef.current) {
      repBottomReachedRef.current = false
      setRepCount((c) => c + 1)
    }
  }

  function handleExerciseChange(exercise: ExerciseType) {
    selectedExerciseRef.current = exercise
    repBottomReachedRef.current = false
    setSelectedExercise(exercise)
    setRepCount(0); setLandmarkCount(0); setPoseVisible(false)
    setAnalysis(createNoPoseAnalysis(exercise))
  }

  // ── Recording ────────────────────────────────────────────────────

  function startRecording() {
    if (cameraStatus !== 'live') return
    framesRef.current = []
    setFrameCount(0)
    setHasRecording(false)
    isRecordingRef.current = true
    setIsRecording(true)
  }

  function stopRecording() {
    if (!isRecordingRef.current) return
    isRecordingRef.current = false
    setIsRecording(false)
    const n = framesRef.current.length
    setHasRecording(n >= 2)
    setFrameCount(n)
  }

  function clearRecording() {
    stopRecording()
    stopPlayback()
    framesRef.current = []
    setFrameCount(0)
    setHasRecording(false)
    const sc = stageCanvasRef.current?.getContext('2d')
    if (sc) { sc.fillStyle = '#07111f'; sc.fillRect(0, 0, stageCanvasRef.current!.width, stageCanvasRef.current!.height) }
  }

  // ── Playback ─────────────────────────────────────────────────────

  function startPlayback() {
    const frames = framesRef.current
    if (frames.length < 2) return
    stopPlayback()

    // Apply symmetric smoothing over the full recording for playback quality
    const smoothedAll = smoothAllFrames(frames.map((f) => f.landmarks), 7)

    isPlayingRef.current = true
    setIsPlaying(true)

    const t0 = frames[0].timestamp
    const wallStart = performance.now()

    const stageCanvas = stageCanvasRef.current
    const sc = stageCanvas?.getContext('2d')
    if (!stageCanvas || !sc) return

    const playFrame = () => {
      if (!isPlayingRef.current) return

      const elapsed = performance.now() - wallStart
      let idx = frames.findIndex((f) => f.timestamp - t0 >= elapsed)
      if (idx === -1) idx = frames.length - 1

      drawPuppet(sc, smoothedAll[idx], stageCanvas.width, stageCanvas.height)

      if (idx >= frames.length - 1) {
        isPlayingRef.current = false
        setIsPlaying(false)
        return
      }
      playAnimFrameRef.current = requestAnimationFrame(playFrame)
    }
    playAnimFrameRef.current = requestAnimationFrame(playFrame)
  }

  function stopPlayback() {
    if (playAnimFrameRef.current !== null) {
      cancelAnimationFrame(playAnimFrameRef.current)
      playAnimFrameRef.current = null
    }
    isPlayingRef.current = false
    setIsPlaying(false)
  }

  const isCameraLive = cameraStatus === 'live'

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <p className="eyebrow">ConHacks 2026 MVP</p>
        <h1>Gym Bro</h1>
        <p className="hero-copy">
          Live pose tracking with GPU-accelerated MediaPipe, procedural puppet rendering,
          7-frame smoothing, timeline recording, and one-click export.
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
                disabled={cameraStatus === 'starting' || modelStatus !== 'ready'}
              >
                {isCameraLive ? 'Stop camera'
                  : cameraStatus === 'starting' ? 'Starting…'
                  : modelStatus === 'loading' ? 'Loading model…'
                  : 'Start camera'}
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => setAudioEnabled((e) => !e)}
              >
                {audioEnabled ? 'Voice on' : 'Voice off'}
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => setSmoothingEnabled((e) => !e)}
              >
                {smoothingEnabled ? 'Smooth on' : 'Smooth off'}
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

          {/* ── Dual canvas row ── */}
          <div className="canvas-row">
            <div className="canvas-panel">
              <span className="canvas-label">MoCap Studio</span>
              <div className="camera-stage">
                <video ref={videoRef} className="camera-layer" playsInline muted />
                <canvas ref={canvasRef} className="overlay-layer" />
                {!isCameraLive && (
                  <div className="stage-overlay">
                    <p>Camera feed</p>
                    <span>
                      {modelStatus === 'loading' ? 'Loading pose model…'
                        : modelStatus === 'error' ? modelError
                        : 'Start camera to begin tracking.'}
                    </span>
                  </div>
                )}
              </div>
            </div>

            <div className="canvas-panel">
              <span className="canvas-label">Stage</span>
              <div className="camera-stage">
                <canvas ref={stageCanvasRef} className="overlay-layer" />
                {!isCameraLive && frameCount === 0 && (
                  <div className="stage-overlay">
                    <p>Puppet view</p>
                    <span>Procedural character driven by pose landmarks — live or playback.</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ── Timeline controls ── */}
          <div className="timeline-bar">
            <div className="timeline-left">
              {!isRecording ? (
                <button
                  type="button"
                  className="record-button"
                  onClick={startRecording}
                  disabled={!isCameraLive}
                >
                  ● Record
                </button>
              ) : (
                <button type="button" className="stop-record-button" onClick={stopRecording}>
                  ■ Stop
                </button>
              )}

              <button
                type="button"
                className="secondary-button"
                onClick={isPlaying ? stopPlayback : startPlayback}
                disabled={!hasRecording || isCameraLive}
              >
                {isPlaying ? '■ Stop playback' : '▶ Play'}
              </button>

              <button
                type="button"
                className="secondary-button"
                onClick={clearRecording}
                disabled={frameCount === 0}
              >
                Clear
              </button>

              <span className="frame-counter">
                {isRecording
                  ? `● REC  ${frameCount} frames`
                  : frameCount > 0
                    ? `${frameCount} frames`
                    : 'No recording'}
              </span>
            </div>

            <div className="timeline-right">
              <button
                type="button"
                className="secondary-button"
                onClick={() => exportJSON(framesRef.current)}
                disabled={!hasRecording}
              >
                JSON
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => exportHTML(framesRef.current)}
                disabled={!hasRecording}
              >
                HTML player
              </button>
            </div>
          </div>
        </div>

        {/* ── Coach panel ── */}
        <aside className="coach-panel">
          <div className="status-grid">
            <article className="status-card">
              <span className="label">Model</span>
              <strong>{modelStatus}</strong>
              <p>
                {modelError || (delegateMode ? `Running on ${delegateMode}` : 'MediaPipe Pose Landmarker lite')}
              </p>
            </article>

            <article className="status-card">
              <span className="label">Camera</span>
              <strong>{cameraStatus}</strong>
              <p>{cameraError || 'Front camera, processed in-browser.'}</p>
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

            <article className="status-card compact">
              <span className="label">Smoothing</span>
              <strong>{smoothingEnabled ? 'on (n=7)' : 'off'}</strong>
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
                <p className="empty-copy">Angles appear once a clear pose is detected.</p>
              )}
            </div>
          </article>

          <article className="notes-card">
            <span className="label">Tips</span>
            <ul>
              <li>Side-on view works best for squats and pushups.</li>
              <li>Keep your full body in frame — head to ankles.</li>
              <li>Record a set, then export the HTML player to share.</li>
            </ul>
          </article>
        </aside>
      </section>
    </main>
  )
}

export default App
