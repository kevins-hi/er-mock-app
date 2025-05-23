// src/components/CheckWork.js
import React, { useEffect, useState, useRef } from "react";

export default function CheckWork({ active, videoRef, canvasRef }) {
  const [showEdges, setShowEdges] = useState(true);
  // const [lineCount, setLineCount] = useState(0);
  const [detections, setDetections] = useState([]);
  const subtractorRef = useRef(null);
  const lineBufferRef = useRef([]); // stores { rho, theta, timestamp }
  const lastDetectionTimeRef = useRef(0);
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  const captureCanvasRef = useRef(null);
  const [capturedImageUrl, setCapturedImageUrl] = useState(null);

  // const leftCanvasRef = useRef(null);
  // const rightCanvasRef = useRef(null);

  const saveCaptureImage = async (cv, src, w, h) => {
    const captureCanvas = captureCanvasRef.current;
    captureCanvas.width = w;
    captureCanvas.height = h;
    cv.imshow(captureCanvas, src);
    const dataUrl = captureCanvas.toDataURL("image/png");
    setCapturedImageUrl(dataUrl);
  };

  useEffect(() => {
    const intervalId = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString());
    }, 1000);

    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (active !== "check" || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    // const leftCanvas = leftCanvasRef.current;
    // const rightCanvas = rightCanvasRef.current;

    const ctx = canvas.getContext("2d");
    let animationId;

    // const updateCanvasPositions = () => {
    //   const rect = canvas.getBoundingClientRect();
    //   Object.assign(leftCanvas.style, {
    //     position: "fixed",
    //     left: `${rect.left - 480}px`,
    //     top: `${rect.top}px`,
    //     width: "480px",
    //     height: "360px",
    //     zIndex: 1000,
    //     transform: "rotateY(180deg)",
    //     WebkitTransform: "rotateY(180deg)",
    //     MozTransform: "rotateY(180deg)"
    //   });
    //   Object.assign(rightCanvas.style, {
    //     position: "fixed",
    //     left: `${rect.left + 480}px`,
    //     top: `${rect.top}px`,
    //     width: "480px",
    //     height: "360px",
    //     zIndex: 1000,
    //     transform: "rotateY(180deg)",
    //     WebkitTransform: "rotateY(180deg)",
    //     MozTransform: "rotateY(180deg)"
    //   });
    // };

    const detectEdges = async () => {
      if (active !== "check") {
        cancelAnimationFrame(animationId);
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const cv = await window.cv;
      if (!cv || video.readyState < 2) {
        animationId = requestAnimationFrame(detectEdges);
        return;
      }

      const w = canvas.width;
      const h = canvas.height;
      video.width = w;
      video.height = h;

      // updateCanvasPositions();

      const src = new cv.Mat(h, w, cv.CV_8UC4);
      const cap = new cv.VideoCapture(video);
      cap.read(src);

      const fgMask = new cv.Mat();
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      if (!subtractorRef.current) {
        subtractorRef.current = new cv.BackgroundSubtractorMOG2(50, 16, false);
      }
      subtractorRef.current.apply(gray, fgMask);

      const lines = new cv.Mat();
      const fgMaskColor = new cv.Mat();
      cv.cvtColor(fgMask, fgMaskColor, cv.COLOR_GRAY2RGBA);

      cv.HoughLines(fgMask, lines, 1, Math.PI / 180, 150);

      const angleThreshold = (15 * Math.PI) / 180;
      const isApproximatelyVerticalOrHorizontal = (theta) => {
        const PI = Math.PI;
        return (
          Math.abs(theta - 0) < angleThreshold ||
          Math.abs(theta - PI / 2) < angleThreshold ||
          Math.abs(theta - PI) < angleThreshold
        );
      };

      const filteredLines = [];
      if (lines.rows < 100) {
        for (let i = 0; i < lines.rows; ++i) {
          let rho = lines.data32F[i * 2];
          let theta = lines.data32F[i * 2 + 1];
          if (isApproximatelyVerticalOrHorizontal(theta)) {
            filteredLines.push({ rho, theta });
          }
        }
      }

      const rhoThreshold = 20;
      const thetaThreshold = (15 * Math.PI) / 180;
      const mergedLines = [];
      for (const line of filteredLines) {
        let merged = false;
        for (const cluster of mergedLines) {
          const avgTheta = cluster.thetaSum / cluster.count;
          const avgRho = cluster.rhoSum / cluster.count;
          if (
            Math.abs(line.theta - avgTheta) < thetaThreshold &&
            Math.abs(line.rho - avgRho) < rhoThreshold
          ) {
            cluster.thetaSum += line.theta;
            cluster.rhoSum += line.rho;
            cluster.count += 1;
            merged = true;
            break;
          }
        }
        if (!merged) {
          mergedLines.push({ thetaSum: line.theta, rhoSum: line.rho, count: 1 });
        }
      }
      // setLineCount(mergedLines.length);

      // === Detection logic ===
      const now = Date.now();
      const threeSecondsAgo = now - 3000; // Time length of buffer

      // Prune old entries
      lineBufferRef.current = lineBufferRef.current.filter(
        line => line.timestamp >= threeSecondsAgo
      );

      let detectionTriggered = false;
      for (const merged of mergedLines) {
        const rho = merged.rhoSum / merged.count;
        const theta = merged.thetaSum / merged.count;

        const similarCount = lineBufferRef.current.filter(
          line =>
            Math.abs(line.rho - rho) < rhoThreshold &&
            Math.abs(line.theta - theta) < thetaThreshold
        ).length;

        if (similarCount >= 30 && now - lastDetectionTimeRef.current > 10000) { // Detect every 10 seconds
          detectionTriggered = true;
          lastDetectionTimeRef.current = now;
          const timestamp = new Date(now).toLocaleTimeString();
          setDetections(prev => [...prev, `Detection at ${timestamp}`]);
          saveCaptureImage(cv, src, w, h);
          break;
        }
      }

      // Add current lines after checking for detection
      for (const merged of mergedLines) {
        const rho = merged.rhoSum / merged.count;
        const theta = merged.thetaSum / merged.count;
        lineBufferRef.current.push({ rho, theta, timestamp: now });
      }

      // Draw merged lines
      for (const cluster of mergedLines) {
        const theta = cluster.thetaSum / cluster.count;
        const rho = cluster.rhoSum / cluster.count;

        const a = Math.cos(theta);
        const b = Math.sin(theta);
        const x0 = a * rho;
        const y0 = b * rho;
        const x1 = Math.round(x0 + 1000 * (-b));
        const y1 = Math.round(y0 + 1000 * a);
        const x2 = Math.round(x0 - 1000 * (-b));
        const y2 = Math.round(y0 - 1000 * a);

        cv.line(
          fgMaskColor,
          new cv.Point(x1, y1),
          new cv.Point(x2, y2),
          new cv.Scalar(0, 0, 255, 255),
          2
        );
      }

      const displayMat = showEdges ? fgMaskColor.clone() : src.clone();
      cv.imshow(canvas, displayMat);
      // cv.imshow(leftCanvas, gray);
      // cv.imshow(rightCanvas, fgMask);

      src.delete();
      gray.delete();
      fgMask.delete();
      fgMaskColor.delete();
      lines.delete();
      displayMat.delete();

      animationId = requestAnimationFrame(detectEdges);
    };

    detectEdges();

    return () => {
      cancelAnimationFrame(animationId);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (subtractorRef.current) {
        subtractorRef.current.delete();
        subtractorRef.current = null;
      }
    };
  }, [active, showEdges, videoRef, canvasRef]);

  if (active !== "check") return null;

  return (
    <div>
      {/* <div style={{ color: "black", marginBottom: "8px" }}>
        Detected Lines: {lineCount}
      </div> */}
      <div className="toggle-threshold" style={{ display: "flex", justifyContent: "center" }}>
        <button
          className="mdc-button mdc-button--raised toggle-button"
          onClick={() => setShowEdges(prev => !prev)}
        >
          <span className="mdc-button__label">
            {showEdges ? "Color View" : "Edge View"}
          </span>
        </button>
      </div>
      <div style={{ color: "black", fontWeight: "bold", marginTop:"20px", marginBottom: "4px" }}>
        Current Time: {currentTime}
      </div>
      <ul style={{ color: "black", marginTop: "8px" }}>
        {detections.slice(-3).map((msg, idx) => (
          <li key={idx}>{msg}</li>
        ))}
      </ul>
      {/* <canvas ref={leftCanvasRef} /> */}
      {/* <canvas ref={rightCanvasRef} /> */}
      {capturedImageUrl && (
        <div style={{ marginTop: "16px", textAlign: "center" }}>
          <img
            src={capturedImageUrl}
            alt="Detection Snapshot"
            style={{ width: "270px", border: "1px solid #ccc" }}
          />
        </div>
      )}
      <canvas ref={captureCanvasRef} style={{ display: "none" }} />
    </div>
  );
}
