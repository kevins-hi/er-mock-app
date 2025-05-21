// src/components/CheckWork.js
import React, { useEffect, useState, useRef } from "react";

export default function CheckWork({ active, videoRef, canvasRef }) {
  const [showEdges, setShowEdges] = useState(false);
  const subtractorRef = useRef(null);

  useEffect(() => {
    if (active !== "check" || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let animationId;

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

      const src = new cv.Mat(h, w, cv.CV_8UC4);
      const cap = new cv.VideoCapture(video);
      cap.read(src);

      const fgMask = new cv.Mat();
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      if (!subtractorRef.current) {
        subtractorRef.current = new cv.BackgroundSubtractorMOG2(500, 16, true);
      }

      subtractorRef.current.apply(gray, fgMask);

      const edges = new cv.Mat();
      cv.Canny(fgMask, edges, 100, 200);

      const edgeColor = new cv.Mat();
      cv.cvtColor(edges, edgeColor, cv.COLOR_GRAY2RGBA);

      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      const displayMat = showEdges ? edgeColor.clone() : src.clone();

      for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);
        if (area > 1000) {
          cv.drawContours(displayMat, contours, i, new cv.Scalar(0, 255, 0, 255), 2);
        }
      }

      cv.imshow(canvas, displayMat);

      // Clean up
      src.delete();
      gray.delete();
      fgMask.delete();
      edges.delete();
      edgeColor.delete();
      contours.delete();
      hierarchy.delete();
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
    <div className="toggle-threshold">
      <button
        className="mdc-button mdc-button--raised toggle-button"
        onClick={() => setShowEdges(prev => !prev)}
      >
        <span className="mdc-button__label">
          {showEdges ? "Color View" : "Edge View"}
        </span>
      </button>
    </div>
  );
}
