import './App.css'
import { usePoseTracking, type Joint } from './hooks/usePoseTracking'

function JointRow({ name, joint }: { name: string; joint: Joint }) {
  const dim = joint.visibility < 0.3
  return (
    <div className={`joint-row ${dim ? 'dim' : ''}`}>
      <span className="joint-name">{name}</span>
      <span className="joint-vals">
        <span title="x (mirrored, 0–1)">x {joint.x.toFixed(3)}</span>
        <span title="y (0–1)">y {joint.y.toFixed(3)}</span>
        <span title="depth">z {joint.z.toFixed(3)}</span>
        <span
          className="vis-badge"
          style={{ opacity: Math.max(0.25, joint.visibility) }}
        >
          {Math.round(joint.visibility * 100)}%
        </span>
      </span>
    </div>
  )
}

export default function App() {
  const {
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
  } = usePoseTracking()

  const isLive    = cameraStatus === 'live'
  const isReady   = modelStatus === 'ready'
  const isLoading = modelStatus === 'loading' || cameraStatus === 'starting'

  return (
    <div className="shell">
      <header className="hero">
        <div className="hero-copy">
          <span className="eyebrow">Live MediaPipe Demo</span>
          <h1>See your pose tracked live in the browser.</h1>
          <p>
            This page opens your camera, runs MediaPipe pose detection in real time,
            and draws the full landmark skeleton directly over the video feed.
          </p>
          <div className="hero-actions">
            <button
              className={isLive ? 'btn-stop' : 'btn-start'}
              onClick={isLive ? stopCamera : startCamera}
              disabled={isLoading || !isReady}
            >
              {isLive ? 'Turn camera off'
                : cameraStatus === 'starting' ? 'Turning camera on…'
                : modelStatus === 'loading' ? 'Loading model…'
                : 'Turn camera on'}
            </button>

            <span className="badge">
              {modelStatus === 'loading' ? 'Loading model…'
                : modelStatus === 'error' ? 'Model error'
                : delegate ? `MediaPipe · ${delegate}`
                : 'Model ready'}
            </span>
          </div>
        </div>

        <div className="hero-notes">
          <div className="note-card">
            <strong>Best results</strong>
            <span>Stand far enough back to fit your full body in frame.</span>
          </div>
          <div className="note-card">
            <strong>Camera control</strong>
            <span>Use the button to switch the live camera feed on or off anytime.</span>
          </div>
        </div>
      </header>

      <div className="workspace">
        <div className="viewer">
          <div className="stage">
            <video ref={videoRef} className="hidden-video" playsInline muted />
            <canvas ref={canvasRef} className="canvas" />

            {!isLive && (
              <div className="stage-placeholder">
                {cameraError
                  ? <><strong>Camera error</strong><span>{cameraError}</span></>
                  : modelError
                    ? <><strong>Model error</strong><span>{modelError}</span></>
                    : <><strong>Camera is off</strong><span>Turn the camera on to start live pose tracking.</span></>}
              </div>
            )}
          </div>

          <div className="viewer-footer">
            <span className={`status-dot ${isLive ? 'live' : ''}`} />
            <span>{isLive ? 'Camera live' : 'Camera off'}</span>
            {poseData && (
              <span className="pose-status">· Pose detected</span>
            )}
            {isLive && !poseData && (
              <span className="pose-status searching">· Searching for pose…</span>
            )}
          </div>
        </div>

        <aside className="data-panel">
          <div className="panel-header">
            <span className="panel-title">Live joint data</span>
            <span className="panel-sub">Mirrored browser coordinates updating frame by frame</span>
          </div>

          {poseData ? (
            <div className="joint-list">
              <JointRow name="Nose"           joint={poseData.nose} />
              <JointRow name="Left shoulder"  joint={poseData.leftShoulder} />
              <JointRow name="Right shoulder" joint={poseData.rightShoulder} />
              <JointRow name="Left elbow"     joint={poseData.leftElbow} />
              <JointRow name="Right elbow"    joint={poseData.rightElbow} />
              <JointRow name="Left wrist"     joint={poseData.leftWrist} />
              <JointRow name="Right wrist"    joint={poseData.rightWrist} />
              <JointRow name="Left hip"       joint={poseData.leftHip} />
              <JointRow name="Right hip"      joint={poseData.rightHip} />
              <JointRow name="Left knee"      joint={poseData.leftKnee} />
              <JointRow name="Right knee"     joint={poseData.rightKnee} />
              <JointRow name="Left ankle"     joint={poseData.leftAnkle} />
              <JointRow name="Right ankle"    joint={poseData.rightAnkle} />

              <div className="timestamp">
                Frame captured at {poseData.capturedAt.toFixed(0)} ms
              </div>
            </div>
          ) : (
            <div className="no-pose">
              {isLive
                ? 'No pose detected — make sure your body is visible.'
                : 'Start the camera to see live joint data.'}
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}
