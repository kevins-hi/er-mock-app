import React, { useEffect, useRef, useState } from "react";
import "./App.css";

// Import MediaPipe Tasks for Vision. You can install this package via npm
// (if available) or load it externally. For this example, we assume it’s installed.
import { FaceLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision"; 

function App() {
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [runningMode, setRunningMode] = useState("VIDEO");
  const [webcamRunning, setWebcamRunning] = useState(true);
  const demosSectionRef = useRef(null);
  const videoRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const webcamButtonRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  // Load and initialize the FaceLandmarker when the component mounts.
  useEffect(() => {
    async function createFaceLandmarker() {
      const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      const fl = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath:
            "models/face_landmarker.task",
          delegate: "GPU",
        },
        outputFaceBlendshapes: false,
        runningMode,
        numFaces: 1,
      });
      setFaceLandmarker(fl);
      // Make the demos section visible once the model is loaded.
      if (demosSectionRef.current) {
        demosSectionRef.current.classList.remove("invisible");
      }
    }
    createFaceLandmarker();
  }, []);

  // Enable or disable the webcam and start predictions.
  const enableCam = async () => {
    if (!faceLandmarker) {
      console.log("Wait! faceLandmarker not loaded yet.");
      return;
    }
    const video = videoRef.current;
    if (webcamButtonRef.current) {
      webcamButtonRef.current.classList.add("removed");
    }

    // Request webcam access.
    const constraints = { video: true };
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
    });
  };

  // Continuously predict using the webcam stream.
  const predictWebcam = async () => {
    const video = videoRef.current;
    const canvasElement = outputCanvasRef.current;
    const canvasCtx = canvasElement.getContext("2d");
    const ratio = video.videoHeight / video.videoWidth;
    video.style.width = "480px";
    video.style.height = 480 * ratio + "px";
    canvasElement.style.width = "480px";
    canvasElement.style.height = 480 * ratio + "px";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    const startTimeMs = performance.now();
    if (lastVideoTimeRef.current !== video.currentTime) {
      lastVideoTimeRef.current = video.currentTime;
      const results = await faceLandmarker.detectForVideo(video, startTimeMs);
      if (results.faceLandmarks) {
        const drawingUtils = new DrawingUtils(canvasCtx);
        results.faceLandmarks.forEach((landmarks) => {
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_TESSELATION,
            { color: "#C0C0C070", lineWidth: 1 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
            { color: "#FF3030" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
            { color: "#FF3030" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
            { color: "#30FF30" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
            { color: "#30FF30" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
            { color: "#E0E0E0" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LIPS,
            { color: "#E0E0E0" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
            { color: "#FF3030" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
            { color: "#30FF30" }
          );
        });
      }
    }
    if (webcamRunning) {
      window.requestAnimationFrame(predictWebcam);
    }
  };

  return (
    <div className="App">
      <h1>
        Face landmark detection using the MediaPipe FaceLandmarker task
      </h1>
      <section id="demos" ref={demosSectionRef} className="invisible">
        <h2>Demo: Webcam continuous face landmarks detection</h2>
        <p>
          Hold your face in front of your webcam to get real-time face
          landmarker detection.
          <br />
          Click <strong>enable webcam</strong> below and grant access to the
          webcam if prompted.
        </p>
        <div id="liveView" className="videoView">
          <button
            ref={webcamButtonRef}
            className="mdc-button mdc-button--raised"
            onClick={enableCam}
          >
            <span className="mdc-button__label">ENABLE WEBCAM</span>
          </button>
          <div style={{ position: "relative" }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              style={{ position: "absolute" }}
            ></video>
            <canvas
              ref={outputCanvasRef}
              className="output_canvas"
              style={{ position: "absolute", left: 0, top: 0 }}
            ></canvas>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
