import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import { FaceLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

function App() {
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [runningMode] = useState("VIDEO");

  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [activeMode, setActiveMode] = useState("default"); // "default", "gaze", or "check"
  const activeModeRef = useRef(activeMode);
  useEffect(() => { activeModeRef.current = activeMode; }, [activeMode]);

  const [tracking, setTracking] = useState(false);
  const [trackingData, setTrackingData] = useState([]);
  const [trackingElapsedTime, setTrackingElapsedTime] = useState(0);
  const trackingBufferRef = useRef([]);
  const lastTrackingTimestampRef = useRef(Date.now());
  const trackingRef = useRef(tracking);
  useEffect(() => { trackingRef.current = tracking; }, [tracking]);

  const videoRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const webcamButtonRef = useRef(null);

  // Coordinate helper
  const relative = (landmark, { width, height }) => [landmark.x * width, landmark.y * height];

  /** Estimate 3D affine transform via least-squares */
  const estimateAffine3DCustom = (cv, src, dst) => {
    const N = src.rows;
    if (N < 4) {
      console.warn("Need at least 4 points to estimate affine transform");
      return { success: false, transformation: null };
    }
    const X = new cv.Mat(N, 4, cv.CV_64F);
    for (let i = 0; i < N; i++) {
      X.data64F[i * 4] = src.data64F[i * 3];
      X.data64F[i * 4 + 1] = src.data64F[i * 3 + 1];
      X.data64F[i * 4 + 2] = src.data64F[i * 3 + 2];
      X.data64F[i * 4 + 3] = 1;
    }
    const Y = dst;

    const Xt = new cv.Mat();
    cv.transpose(X, Xt);

    const XtX = new cv.Mat();
    cv.gemm(Xt, X, 1, new cv.Mat(), 0, XtX);

    const XtX_inv = new cv.Mat();
    cv.invert(XtX, XtX_inv, cv.DECOMP_SVD);

    const XtY = new cv.Mat();
    cv.gemm(Xt, Y, 1, new cv.Mat(), 0, XtY);

    const A = new cv.Mat();
    cv.gemm(XtX_inv, XtY, 1, new cv.Mat(), 0, A);

    const T = new cv.Mat();
    cv.transpose(A, T);

    [X, Xt, XtX, XtX_inv, XtY, A].forEach((m) => m.delete());
    return { success: true, transformation: T };
  };

  // Draw gaze based on landmarks
  const drawGaze = async (frame, landmarks, ctx) => {
    if (activeModeRef.current !== "gaze") return;
    const cv = await window.cv;
    if (!cv) return;
    const frameShape = { width: frame.videoWidth, height: frame.videoHeight };

    // 2D & 3D image points for pose estimation
    const imgPoints2D = [4,152,263,33,287,57].map(i => relative(landmarks[i], frameShape));
    const imgPoints3D = imgPoints2D.map(([x,y]) => [x,y,0]);
    const flat2D = imgPoints2D.flat();
    const flat3D = imgPoints3D.flat();

    const mat2D = cv.matFromArray(6, 2, cv.CV_64F, flat2D);
    const mat3D = cv.matFromArray(6, 3, cv.CV_64F, flat3D);

    const modelPts = cv.matFromArray(6, 3, cv.CV_64F, [
      0,0,0,    0,-63.6,-12.5,
      -43.3,32.7,-26, 43.3,32.7,-26,
      -28.9,-28.9,-24.1, 28.9,-28.9,-24.1
    ]);
    const eyeCenterL = cv.matFromArray(3,1,cv.CV_64F,[29.05,32.7,-39.5]);
    const eyeCenterR = cv.matFromArray(3,1,cv.CV_64F,[-29.05,32.7,-39.5]);

    const focal = frameShape.width;
    const center = [focal/2, frameShape.height/2];
    const cameraM = cv.matFromArray(3,3,cv.CV_64F,[
      focal,0,center[0], 0,focal,center[1], 0,0,1
    ]);
    const dist = cv.Mat.zeros(4,1,cv.CV_64F);
    const rvec = new cv.Mat(), tvec = new cv.Mat();
    cv.solvePnP(modelPts, mat2D, cameraM, dist, rvec, tvec, false, cv.SOLVEPNP_ITERATIVE);

    const [lx,ly] = relative(landmarks[468], frameShape);
    const [rx,ry] = relative(landmarks[473], frameShape);

    const { success, transformation } = estimateAffine3DCustom(cv, mat3D, modelPts);
    let gazeL, gazeR;
    if (success) {
      const compute = (px,py,center3) => {
        const vec = cv.matFromArray(4,1,cv.CV_64F,[px,py,0,1]);
        const world = new cv.Mat();
        cv.gemm(transformation, vec, 1, new cv.Mat(), 0, world);
        const world3 = world.rowRange(0,3);
        const diff = new cv.Mat();
        cv.subtract(world3, center3, diff);
        const scale = new cv.Mat(diff.rows,diff.cols,diff.type());
        scale.setTo(new cv.Scalar(10));
        const scaled = new cv.Mat();
        cv.multiply(diff, scale, scaled);
        const endpoint = new cv.Mat();
        cv.add(center3, scaled, endpoint);
        const proj = new cv.Mat();
        cv.projectPoints(endpoint, rvec, tvec, cameraM, dist, proj);
        const head = cv.matFromArray(3,1,cv.CV_64F,[world.data64F[0],world.data64F[1],40]);
        const hproj = new cv.Mat();
        cv.projectPoints(head, rvec, tvec, cameraM, dist, hproj);
        const gaze = [
          px + (proj.data64F[0]-px) - (hproj.data64F[0]-px),
          py + (proj.data64F[1]-py) - (hproj.data64F[1]-py)
        ];
        [vec,world,world3,diff,scale,scaled,endpoint,proj,head,hproj].forEach(m=>m.delete());
        return gaze;
      };
      gazeL = compute(lx,ly,eyeCenterL);
      gazeR = compute(rx,ry,eyeCenterR);

      [[lx,ly,gazeL],[rx,ry,gazeR]].forEach(([sx,sy,[gx,gy]]) => {
        ctx.beginPath(); ctx.moveTo(sx,sy); ctx.lineTo(gx,gy);
        ctx.strokeStyle = "blue"; ctx.lineWidth = 2; ctx.stroke();
      });

      transformation.delete();
    }

    const avgP = [(lx+rx)/2,(ly+ry)/2];
    const avgG = [(gazeL[0]+gazeR[0])/2,(gazeL[1]+gazeR[1])/2];
    const vec = [avgG[0]-avgP[0],avgG[1]-avgP[1]];
    const mag = Math.hypot(...vec);
    const isLooking = mag < 60;
    document.body.style.backgroundColor = isLooking ? "white" : "red";

    ctx.save();
    ctx.setTransform(-1,0,0,1,ctx.canvas.width,0);
    ctx.font = "24px Arial";
    ctx.fillStyle = "blue";
    ctx.fillText(`Looking: ${isLooking}`, ctx.canvas.width - 390, 30);
    ctx.restore();

    if (trackingRef.current) {
      const now = Date.now();
      trackingBufferRef.current.push(isLooking);
      if (now - lastTrackingTimestampRef.current >= 1000) {
        const votes = trackingBufferRef.current;
        const majority = votes.filter(Boolean).length > votes.length / 2;
        setTrackingData(prev => [...prev, { isLooking: majority, timestamp: new Date().toLocaleTimeString() }]);
        trackingBufferRef.current = [];
        lastTrackingTimestampRef.current = now;
      }
    }

    [mat2D, mat3D, modelPts, eyeCenterL, eyeCenterR, cameraM, dist, rvec, tvec].forEach(m => m.delete());
  };

  // Initialize FaceLandmarker
  useEffect(() => {
    (async () => {
      const resolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      const fl = await FaceLandmarker.createFromOptions(resolver, {
        baseOptions: { modelAssetPath: "models/face_landmarker.task", delegate: "GPU" },
        outputFaceBlendshapes: false,
        runningMode,
        numFaces: 1
      });
      setFaceLandmarker(fl);
    })();
  }, [runningMode]);

  // Enable webcam and start
  const enableCam = () => {
    if (!faceLandmarker) return console.warn("FaceLandmarker not loaded");
    webcamButtonRef.current?.classList.add("removed");
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
      videoRef.current.srcObject = stream;
      videoRef.current.addEventListener("loadeddata", predictWebcam);
      setCameraEnabled(true);
    });
  };

  // Toggle tracking
  const toggleTracking = () => {
    if (!tracking) {
      trackingBufferRef.current = [];
      lastTrackingTimestampRef.current = Date.now();
      setTrackingData([]);
    }
    setTracking(prev => !prev);
  };

  // Timer for tracking elapsed time
  useEffect(() => {
    let interval;
    if (tracking) {
      const start = Date.now();
      interval = setInterval(() => setTrackingElapsedTime(Date.now() - start), 1000);
    } else {
      setTrackingElapsedTime(0);
    }
    return () => clearInterval(interval);
  }, [tracking]);

  // Predict loop
  const predictWebcam = async () => {
    const video = videoRef.current;
    const canvas = outputCanvasRef.current;
    const ctx = canvas.getContext("2d");
    const ratio = video.videoHeight / video.videoWidth;
    const width = 480;
    video.style.width = `${width}px`;
    video.style.height = `${width * ratio}px`;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${width * ratio}px`;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const results = await faceLandmarker.detectForVideo(video, performance.now());
    if (results.faceLandmarks && activeModeRef.current === "gaze") {
      const utils = new DrawingUtils(ctx);
      results.faceLandmarks.forEach(landmarks => {
        utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
        utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
        utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
        utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
        utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
        utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
        utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
        utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
        utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });
        drawGaze(video, landmarks, ctx);
      });
    }
    window.requestAnimationFrame(predictWebcam);
  };

  return (
    <div className="App">
      {!cameraEnabled ? (
        <div className="controls">
          <button ref={webcamButtonRef} className="mdc-button mdc-button--raised webcamButton" onClick={enableCam}>
            <span className="mdc-button__label">ENABLE WEBCAM</span>
          </button>
        </div>
      ) : (
        <div className="controls">
          <button className="mdc-button mdc-button--raised" onClick={() => setActiveMode("default")}>        
            <span className="mdc-button__label">{activeMode === "default" ? "✔ Default View" : "Default View"}</span>
          </button>
          <button className="mdc-button mdc-button--raised" onClick={() => setActiveMode("gaze")}>        
            <span className="mdc-button__label">{activeMode === "gaze" ? "✔ Gaze Tracking" : "Gaze Tracking"}</span>
          </button>
          <button className="mdc-button mdc-button--raised" onClick={() => setActiveMode("check")}>        
            <span className="mdc-button__label">{activeMode === "check" ? "✔ Check Work" : "Check Work"}</span>
          </button>
        </div>
      )}

      <div className="videoView">
        <div style={{ position: "relative" }}>
          <video ref={videoRef} autoPlay playsInline style={{ position: "absolute" }} />
          <canvas ref={outputCanvasRef} className="output_canvas" style={{ position: "absolute", left: 0, top: 0 }} />
        </div>
      </div>

      {cameraEnabled && activeMode === "gaze" && (
        <div className="tracking-section">
          <button onClick={toggleTracking} className="mdc-button mdc-button--raised trackingButton">
            <span className="mdc-button__label">
              {tracking ? `Stop Tracking (${Math.floor(trackingElapsedTime / 1000)}s)` : "Start Tracking"}
            </span>
          </button>
          {!tracking && trackingData.length > 0 && (
            <div className="tracking-data">
              <h3>Tracking Data</h3>
              <ul>
                {trackingData.map((item, index) => (
                  <li key={index}>{item.timestamp}: {item.isLooking ? "Looking" : "Not Looking"}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
