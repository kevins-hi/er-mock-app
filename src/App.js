import React, { useEffect, useRef, useState } from "react";
import "./App.css";

// Import MediaPipe Tasks for Vision.
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

  // -----------------------------
  // Helper functions for coordinate conversion
  // -----------------------------
  const relative = (landmark, frameShape) => {
    return [landmark.x * frameShape.width, landmark.y * frameShape.height];
  };

  const relativeT = (landmark, frameShape) => {
    return [landmark.x * frameShape.width, landmark.y * frameShape.height, 0];
  };

  /**
   * Custom least-squares solution to estimate a 3D affine transform.
   *
   * @param {cv.Mat} src - Source 3D points (N x 3, CV_64F).
   * @param {cv.Mat} dst - Destination 3D points (N x 3, CV_64F).
   * @returns {{success: boolean, transformation: cv.Mat|null}} - Object containing the success flag and the transformation (3 x 4) if successful.
   */
  const estimateAffine3DCustom = (cv, src, dst) => {
    // Ensure there are enough points (at least 4) to compute a robust affine transform.
    let N = src.rows;
    if (N < 4) {
      console.log("Need at least 4 points to estimate an affine transform");
      return { success: false, transformation: null };
    }

    // Create matrix X of size (N x 4): each row is [x, y, z, 1] from src.
    let X = new cv.Mat(N, 4, cv.CV_64F);
    for (let i = 0; i < N; i++) {
      X.data64F[i * 4 + 0] = src.data64F[i * 3 + 0];
      X.data64F[i * 4 + 1] = src.data64F[i * 3 + 1];
      X.data64F[i * 4 + 2] = src.data64F[i * 3 + 2];
      X.data64F[i * 4 + 3] = 1.0;
    }

    // Y is the destination matrix (N x 3). We assume dst is already of correct size and type.
    let Y = dst; // (N x 3)

    // Compute Xᵀ (transpose of X)
    let Xt = new cv.Mat();
    cv.transpose(X, Xt);

    // Compute XtX = Xᵀ * X (4 x 4 matrix)
    let XtX = new cv.Mat();
    cv.gemm(Xt, X, 1, new cv.Mat(), 0, XtX);

    // Invert XtX (using SVD for numerical stability)
    let XtX_inv = new cv.Mat();
    cv.invert(XtX, XtX_inv, cv.DECOMP_SVD);

    // Compute XtY = Xᵀ * Y (4 x 3 matrix)
    let XtY = new cv.Mat();
    cv.gemm(Xt, Y, 1, new cv.Mat(), 0, XtY);

    // Solve for A: A = (XᵀX)⁻¹ * (XᵀY), A will be a 4 x 3 matrix.
    let A = new cv.Mat();
    cv.gemm(XtX_inv, XtY, 1, new cv.Mat(), 0, A);

    // The affine transformation T is the transpose of A, making T a 3 x 4 matrix.
    let T = new cv.Mat();
    cv.transpose(A, T);

    // Clean up temporary matrices.
    X.delete(); Xt.delete(); XtX.delete(); XtX_inv.delete(); XtY.delete(); A.delete();
    return { success: true, transformation: T };
  }

  // -----------------------------
  // Draw gaze direction based on landmarks.
  // This function converts the Python logic to JS using OpenCV.js.
  // -----------------------------
  const drawGaze = async (frame, landmarks, canvasCtx) => {
    // Use window.cv since OpenCV.js attaches to window.
    const cv = await window.cv;
    if (!cv) return; // Extra safety check

    const frameShape = { width: frame.videoWidth, height: frame.videoHeight };

    // Prepare 2D image points (for solvePnP)
    const imagePoints = [
      relative(landmarks[4], frameShape),    // Nose tip
      relative(landmarks[152], frameShape),    // Chin
      relative(landmarks[263], frameShape),    // Left eye left corner
      relative(landmarks[33], frameShape),     // Right eye right corner
      relative(landmarks[287], frameShape),    // Left Mouth corner
      relative(landmarks[57], frameShape)      // Right mouth corner
    ];

    // Prepare image points with zero z-value (for estimateAffine3D)
    const imagePoints1 = [
      relativeT(landmarks[4], frameShape),
      relativeT(landmarks[152], frameShape),
      relativeT(landmarks[263], frameShape),
      relativeT(landmarks[33], frameShape),
      relativeT(landmarks[287], frameShape),
      relativeT(landmarks[57], frameShape)
    ];

    const flattenPoints = (points) =>
      points.reduce((acc, val) => acc.concat(val), []);

    let imagePointsMat = cv.matFromArray(6, 2, cv.CV_64F, flattenPoints(imagePoints));
    let imagePoints1Mat = cv.matFromArray(6, 3, cv.CV_64F, flattenPoints(imagePoints1));

    let modelPoints = cv.matFromArray(6, 3, cv.CV_64F, [
      0.0, 0.0, 0.0,         // Nose tip
      0.0, -63.6, -12.5,      // Chin
      -43.3, 32.7, -26,       // Left eye, left corner
      43.3, 32.7, -26,        // Right eye, right corner
      -28.9, -28.9, -24.1,     // Left Mouth corner
      28.9, -28.9, -24.1       // Right mouth corner
    ]);

    let eyeBallCenterRight = cv.matFromArray(3, 1, cv.CV_64F, [-29.05, 32.7, -39.5]);
    let eyeBallCenterLeft = cv.matFromArray(3, 1, cv.CV_64F, [29.05, 32.7, -39.5]);

    let focalLength = frameShape.width;
    let center = [frameShape.width / 2, frameShape.height / 2];
    let cameraMatrix = cv.matFromArray(3, 3, cv.CV_64F, [
      focalLength, 0, center[0],
      0, focalLength, center[1],
      0, 0, 1
    ]);

    let distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64F);

    let rvec = new cv.Mat();
    let tvec = new cv.Mat();
    cv.solvePnP(modelPoints, imagePointsMat, cameraMatrix, distCoeffs, rvec, tvec, false, cv.SOLVEPNP_ITERATIVE);

    // Get left pupil coordinates (using landmark index 468)
    const leftPupil = relative(landmarks[468], frameShape);

    const { success, transformation } = estimateAffine3DCustom(cv, imagePoints1Mat, modelPoints);
    if (success) {
      let pupilVec = cv.matFromArray(4, 1, cv.CV_64F, [leftPupil[0], leftPupil[1], 0, 1]);
      let pupilWorld = new cv.Mat();
      cv.gemm(transformation, pupilVec, 1, new cv.Mat(), 0, pupilWorld);

      let diff = new cv.Mat();
      let pupilWorld3 = pupilWorld.rowRange(0, 3);
      cv.subtract(pupilWorld3, eyeBallCenterLeft, diff);

      let scaledDiff = new cv.Mat();
      let broadcastMat = new cv.Mat(diff.rows, diff.cols, diff.type());
      broadcastMat.setTo(new cv.Scalar(10));
      cv.multiply(diff, broadcastMat, scaledDiff);
      broadcastMat.delete();

      let S = new cv.Mat();
      cv.add(eyeBallCenterLeft, scaledDiff, S);

      let S_point = new cv.Mat();
      cv.projectPoints(S, rvec, tvec, cameraMatrix, distCoeffs, S_point);
      let eyePupil2D = [S_point.data64F[0], S_point.data64F[1]];

      let headPoint = cv.matFromArray(3, 1, cv.CV_64F, [pupilWorld.data64F[0], pupilWorld.data64F[1], 40]);
      let headPose = new cv.Mat();
      cv.projectPoints(headPoint, rvec, tvec, cameraMatrix, distCoeffs, headPose);
      let headPose2D = [headPose.data64F[0], headPose.data64F[1]];

      let gaze = [
        leftPupil[0] + (eyePupil2D[0] - leftPupil[0]) - (headPose2D[0] - leftPupil[0]),
        leftPupil[1] + (eyePupil2D[1] - leftPupil[1]) - (headPose2D[1] - leftPupil[1])
      ];

      // Draw the gaze line.
      canvasCtx.beginPath();
      canvasCtx.moveTo(leftPupil[0], leftPupil[1]);
      canvasCtx.lineTo(gaze[0], gaze[1]);
      canvasCtx.strokeStyle = "blue";
      canvasCtx.lineWidth = 2;
      canvasCtx.stroke();

      pupilVec.delete();
      pupilWorld.delete();
      diff.delete();
      scaledDiff.delete();
      S.delete();
      S_point.delete();
      headPoint.delete();
      headPose.delete();
    }

    imagePointsMat.delete();
    imagePoints1Mat.delete();
    modelPoints.delete();
    eyeBallCenterRight.delete();
    eyeBallCenterLeft.delete();
    cameraMatrix.delete();
    distCoeffs.delete();
    rvec.delete();
    tvec.delete();
    transformation.delete();
  };

  // -----------------------------
  // Load and initialize the FaceLandmarker when the component mounts.
  // -----------------------------
  useEffect(() => {
    async function createFaceLandmarker() {
      const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      const fl = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: "models/face_landmarker.task",
          delegate: "GPU",
        },
        outputFaceBlendshapes: false,
        runningMode,
        numFaces: 1,
      });
      setFaceLandmarker(fl);
      if (demosSectionRef.current) {
        demosSectionRef.current.classList.remove("invisible");
      }
    }
    createFaceLandmarker();
  }, [runningMode]);

  // -----------------------------
  // Enable webcam and start predictions.
  // -----------------------------
  const enableCam = async () => {
    if (!faceLandmarker) {
      console.log("Wait! faceLandmarker not loaded yet.");
      return;
    }
    const video = videoRef.current;
    if (webcamButtonRef.current) {
      webcamButtonRef.current.classList.add("removed");
    }
    const constraints = { video: true };
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
    });
  };

  // -----------------------------
  // Continuously predict using the webcam stream.
  // -----------------------------
  const predictWebcam = async () => {
    // console.log((await window.cv).projectPoints);
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

          // Gaze estimation
          drawGaze(video, landmarks, canvasCtx);
        });
      }
    }
    if (webcamRunning) {
      window.requestAnimationFrame(predictWebcam);
    }
  };

  return (
    <div className="App">
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
