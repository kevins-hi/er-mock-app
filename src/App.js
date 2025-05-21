// src/App.js
import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import GazeTracking from "./components/GazeTracking";
import CheckWork from "./components/CheckWork";

function App() {
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [runningMode] = useState("VIDEO");

  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [activeMode, setActiveMode] = useState("default"); // "default", "gaze", or "check"
  const activeModeRef = useRef(activeMode);
  useEffect(() => { activeModeRef.current = activeMode; }, [activeMode]);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const webcamButtonRef = useRef(null);

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
      videoRef.current.addEventListener("loadeddata", () => {});
      setCameraEnabled(true);
    });
  };

  useEffect(() => {
    if (!cameraEnabled || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    const resize = () => {
      const ratio = video.videoHeight / video.videoWidth;
      const width = 480;
      video.style.width = `${width}px`;
      video.style.height = `${width * ratio}px`;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${width * ratio}px`;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    };

    if (video.readyState >= 2) {
      resize();
    } else {
      video.addEventListener("loadeddata", resize, { once: true });
    }
  }, [cameraEnabled]);

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
          <canvas ref={canvasRef} style={{ position: "absolute", left: 0, top: 0 }} />
        </div>
      </div>

      <GazeTracking
        active={activeMode}
        activeRef={activeModeRef}
        faceLandmarker={faceLandmarker}
        videoRef={videoRef}
        canvasRef={canvasRef}
      />
      <CheckWork
        active={activeMode}
        videoRef={videoRef}
        canvasRef={canvasRef}
      />
    </div>
  );
}

export default App;
