import React, { useEffect, useRef, useState } from 'react';
import './App.css';
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

function App() {
  const demosRef = useRef(null);
  const videoRef = useRef(null);
  const webcamButtonRef = useRef(null);
  const [faceDetector, setFaceDetector] = useState(null);
  const [runningMode, setRunningMode] = useState("IMAGE");
  const childrenRef = useRef([]); // used for webcam overlays

  // Initialize the face detector on mount.
  useEffect(() => {
    const initializeFaceDetector = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
      );
      const detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
          delegate: "GPU"
        },
        runningMode: runningMode
      });
      setFaceDetector(detector);
      if (demosRef.current) {
        demosRef.current.classList.remove("invisible");
      }
    };
    initializeFaceDetector();
  }, [runningMode]);

  // Handle image click for detection
  const handleImageClick = async (e) => {
    if (!faceDetector) {
      console.log("Face detector not loaded yet");
      return;
    }
    const container = e.currentTarget;
    // Remove any previous overlays from this container
    ['highlighter', 'info', 'key-point'].forEach(cls => {
      const elements = container.querySelectorAll(`.${cls}`);
      elements.forEach(el => el.remove());
    });
    // If webcam mode was active, switch back to IMAGE mode.
    if (runningMode === "VIDEO") {
      setRunningMode("IMAGE");
      await faceDetector.setOptions({ runningMode: "IMAGE" });
    }
    const img = container.querySelector("img");
    const ratio = img.height / img.naturalHeight;
    const detections = (await faceDetector.detect(img)).detections;
    console.log(detections);
    displayImageDetections(detections, img, container, ratio);
  };

  const displayImageDetections = (detections, img, container, ratio) => {
    detections.forEach(detection => {
      // Create info text element.
      const infoP = document.createElement("p");
      infoP.className = "info";
      infoP.innerText =
        "Confidence: " +
        Math.round(parseFloat(detection.categories[0].score) * 100) +
        "% .";
      infoP.style.left = detection.boundingBox.originX * ratio + "px";
      infoP.style.top = (detection.boundingBox.originY * ratio - 30) + "px";
      infoP.style.width = (detection.boundingBox.width * ratio - 10) + "px";
      infoP.style.height = "20px";

      // Create highlighter overlay.
      const highlighter = document.createElement("div");
      highlighter.className = "highlighter";
      highlighter.style.left = detection.boundingBox.originX * ratio + "px";
      highlighter.style.top = detection.boundingBox.originY * ratio + "px";
      highlighter.style.width = detection.boundingBox.width * ratio + "px";
      highlighter.style.height = detection.boundingBox.height * ratio + "px";

      container.appendChild(highlighter);
      container.appendChild(infoP);

      // Draw keypoints.
      detection.keypoints.forEach(keypoint => {
        const keypointEl = document.createElement("span");
        keypointEl.className = "key-point";
        keypointEl.style.top = (keypoint.y * img.height - 3) + "px";
        keypointEl.style.left = (keypoint.x * img.width - 3) + "px";
        container.appendChild(keypointEl);
      });
    });
  };

  // Enable webcam detection.
  const enableCam = async () => {
    if (!faceDetector) {
      alert("Face Detector is still loading. Please try again..");
      return;
    }
    if (webcamButtonRef.current) {
      webcamButtonRef.current.classList.add("removed");
    }
    const constraints = { video: true };
    navigator.mediaDevices
      .getUserMedia(constraints)
      .then(function (stream) {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.addEventListener("loadeddata", predictWebcam);
        }
      })
      .catch(err => console.error(err));
  };

  let lastVideoTime = -1;
  const predictWebcam = async () => {
    if (!faceDetector) return;
    if (runningMode === "IMAGE") {
      setRunningMode("VIDEO");
      await faceDetector.setOptions({ runningMode: "VIDEO" });
    }
    const startTimeMs = performance.now();
    if (videoRef.current.currentTime !== lastVideoTime) {
      lastVideoTime = videoRef.current.currentTime;
      const detections = (await faceDetector.detectForVideo(videoRef.current, startTimeMs)).detections;
      displayVideoDetections(detections);
    }
    window.requestAnimationFrame(predictWebcam);
  };

  const displayVideoDetections = (detections) => {
    const liveView = document.getElementById("liveView");
    // Remove any previous overlays.
    childrenRef.current.forEach(child => liveView.removeChild(child));
    childrenRef.current = [];

    detections.forEach(detection => {
      const infoP = document.createElement("p");
      infoP.innerText =
        "Confidence: " +
        Math.round(parseFloat(detection.categories[0].score) * 100) +
        "% .";
      infoP.style.left =
        (videoRef.current.offsetWidth - detection.boundingBox.width - detection.boundingBox.originX) + "px";
      infoP.style.top = (detection.boundingBox.originY - 30) + "px";
      infoP.style.width = (detection.boundingBox.width - 10) + "px";

      const highlighter = document.createElement("div");
      highlighter.className = "highlighter";
      highlighter.style.left =
        (videoRef.current.offsetWidth - detection.boundingBox.width - detection.boundingBox.originX) + "px";
      highlighter.style.top = detection.boundingBox.originY + "px";
      highlighter.style.width = (detection.boundingBox.width - 10) + "px";
      highlighter.style.height = detection.boundingBox.height + "px";

      liveView.appendChild(highlighter);
      liveView.appendChild(infoP);
      childrenRef.current.push(highlighter, infoP);

      detection.keypoints.forEach(keypoint => {
        const keypointEl = document.createElement("span");
        keypointEl.className = "key-point";
        keypointEl.style.top = (keypoint.y * videoRef.current.offsetHeight - 3) + "px";
        keypointEl.style.left =
          (videoRef.current.offsetWidth - keypoint.x * videoRef.current.offsetWidth - 3) + "px";
        liveView.appendChild(keypointEl);
        childrenRef.current.push(keypointEl);
      });
    });
  };

  return (
    <div className="App">
      <h1>Face detection using the MediaPipe Face Detector task</h1>
      <section id="demos" ref={demosRef} className="invisible">
        <h2>Demo: Webcam continuous face detection</h2>
        <p>
          Detect faces from your webcam. When ready click "enable webcam" below
          and accept access to the webcam.
        </p>
        <div id="liveView" className="videoView">
          <button
            id="webcamButton"
            ref={webcamButtonRef}
            className="mdc-button mdc-button--raised"
            onClick={enableCam}
          >
            <span className="mdc-button__ripple"></span>
            <span className="mdc-button__label">ENABLE WEBCAM</span>
          </button>
          <video id="webcam" ref={videoRef} autoPlay playsInline></video>
        </div>
      </section>
    </div>
  );
}

export default App;
