// src/components/CheckWork.js
import React, { useEffect, useState, useRef } from "react";
import { AzureOpenAI } from "openai";
import { z } from "zod";
import { zodTextFormat } from "openai/helpers/zod";

export const PaperCheck = z.object({
  valid: z.boolean(),
});

export const PaperContent = z.object({
  content: z.string(),
});

export async function checkIfHoldingPaper(client, base64Image) {
  const response = await client.responses.parse({
    model: "gpt-4.1-nano",
    input: [
      { role: "system", content: "Determine if this image contains someone holding a piece of paper. Output true or false as 'valid'." },
      {
        role: "user",
        content: [
          { type: "input_text", text: "Is this a photo of someone holding up a piece of paper?" },
          {
            type: "input_image",
            image_url: base64Image,
          },
        ],
      },
    ],
    text: {
      format: zodTextFormat(PaperCheck, "paperCheck"),
    },
  });

  return response.output_parsed.valid;
}

export async function parsePaperContent(client, base64Image) {
  const response = await client.responses.parse({
    model: "gpt-4.1",
    input: [
      { role: "system", content: "Determine what is written on the paper. Output the content as 'content'." },
      {
        role: "user",
        content: [
          { type: "input_text", text: "What is written on the paper?" },
          {
            type: "input_image",
            image_url: base64Image,
          },
        ],
      },
    ],
    text: {
      format: zodTextFormat(PaperContent, "paperContent"),
    },
  });

  return response.output_parsed.content;
}

export default function CheckWork({ active, videoRef, canvasRef }) {
  const client = new AzureOpenAI({ 
    endpoint: process.env.REACT_APP_AZURE_OPENAI_ENDPOINT,
    apiKey: process.env.REACT_APP_AZURE_OPENAI_API_KEY,
    apiVersion: "2025-03-01-preview",
    dangerouslyAllowBrowser: true 
  });

  const [showEdges, setShowEdges] = useState(true);
  const [detections, setDetections] = useState([]);
  const subtractorRef = useRef(null);
  const lineBufferRef = useRef([]); // stores { rho, theta, timestamp }
  const lastDetectionTimeRef = useRef(0);
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  const captureCanvasRef = useRef(null);
  const [capturedImageUrl, setCapturedImageUrl] = useState(null);
  const [parsedContent, setParsedContent] = useState(null);
  const isValidationInProgressRef = useRef(false);

  const saveCaptureImage = async (cv, src, w, h) => {
    setParsedContent("validating...");

    const captureCanvas = captureCanvasRef.current;
    captureCanvas.width = w;
    captureCanvas.height = h;
    cv.imshow(captureCanvas, src);
    const dataUrl = captureCanvas.toDataURL("image/png");
    setCapturedImageUrl(dataUrl);

    // GPT check
    const isValid = await checkIfHoldingPaper(client, dataUrl);

    // Update detections state
    setDetections(prev => {
      // Update only the last detection
      const updated = [...prev];
      updated[updated.length - 1] = { ...updated[updated.length - 1], valid: isValid };
      return updated;
    });
    
    // Reset validation flag
    isValidationInProgressRef.current = false;

    if (isValid) {
      setParsedContent("parsing...");
      const content = await parsePaperContent(client, dataUrl);
      setParsedContent(content);
    } else {
      setParsedContent("No paper detected");
    }
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
        subtractorRef.current = new cv.BackgroundSubtractorMOG2(50, 16, false);
      }
      subtractorRef.current.apply(gray, fgMask);

      // Dilation increases false positives
      // const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
      // cv.dilate(fgMask, fgMask, kernel);

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

        if (similarCount >= 30 && now - lastDetectionTimeRef.current > 3000 && !isValidationInProgressRef.current) { // Cooldown
          detectionTriggered = true;
          lastDetectionTimeRef.current = now;
          isValidationInProgressRef.current = true;
          const timestamp = new Date(now).toLocaleTimeString();
          const newEntry = { message: `Detection at ${timestamp}`, valid: null };
          setDetections(prev => [...prev, newEntry]);
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
        {detections.slice(-3).map((entry, idx) => (
          <li key={idx}>
            {entry.message}
            {entry.valid === true ? " (✅ Valid)" : entry.valid === false ? " (❌ Invalid)" : " (⚠️ pending)"}
          </li>
        ))}
      </ul>
      {capturedImageUrl && (
        <div style={{ marginTop: "16px", display: "flex", alignItems: "flex-start", justifyContent: "center", gap: "16px" }}>
          <img
            src={capturedImageUrl}
            alt="Detection Snapshot"
            style={{ width: "270px", border: "1px solid #ccc" }}
          />
          {parsedContent && (
            <div style={{ 
              maxWidth: "300px", 
              padding: "12px", 
              border: "1px solid #ddd", 
              borderRadius: "4px", 
              backgroundColor: "#f9f9f9",
              color: "black"
            }}>
              <h4 style={{ margin: "0 0 8px 0", fontSize: "14px", fontWeight: "bold" }}>Paper Content:</h4>
              <p style={{ margin: "0", fontSize: "12px", lineHeight: "1.4" }}>{parsedContent}</p>
            </div>
          )}
        </div>
      )}
      <canvas ref={captureCanvasRef} style={{ display: "none" }} />
    </div>
  );
}
