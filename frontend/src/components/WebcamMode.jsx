import { useEffect, useRef, useState } from "react";
import { predict } from "../api.js";
import { drawDetections } from "../draw.js";

const CAPTURE_WIDTH = 640; // image reduite envoyee a l'API : plus rapide
const INTERVAL_MS = 100; // environ 10 images par seconde au maximum

function toJpeg(canvas) {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
}

export default function WebcamMode({ onResult, onError, disabled }) {
  const [running, setRunning] = useState(false);
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(document.createElement("canvas"));
  const streamRef = useRef(null);
  const onResultRef = useRef(onResult);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onResultRef.current = onResult;
    onErrorRef.current = onError;
  }, [onResult, onError]);

  useEffect(() => {
    if (!running) return undefined;
    let active = true;
    let timer = null;

    async function loop() {
      if (!active) return;
      const video = videoRef.current;
      if (video && video.readyState >= 2) {
        const capture = captureRef.current;
        capture.width = CAPTURE_WIDTH;
        capture.height = Math.round((video.videoHeight / video.videoWidth) * CAPTURE_WIDTH);
        capture.getContext("2d").drawImage(video, 0, 0, capture.width, capture.height);
        try {
          const result = await predict(await toJpeg(capture));
          if (!active) return;
          const overlay = overlayRef.current;
          overlay.width = capture.width;
          overlay.height = capture.height;
          drawDetections(overlay.getContext("2d"), result.detections, { mirrored: true });
          onResultRef.current(result);
        } catch (err) {
          onErrorRef.current(err.message);
        }
      }
      timer = setTimeout(loop, INTERVAL_MS); // une seule requete a la fois
    }

    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        loop();
      } catch {
        onErrorRef.current("Impossible d'ouvrir la webcam (autorisation refusée ?).");
        setRunning(false);
      }
    }

    start();
    return () => {
      active = false;
      clearTimeout(timer);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [running]);

  return (
    <div className="mode">
      <div className="toolbar">
        <button
          className={`button ${running ? "button--stop" : ""}`}
          disabled={disabled && !running}
          onClick={() => setRunning((r) => !r)}
        >
          {running ? "Arrêter la webcam" : "Démarrer la webcam"}
        </button>
        <span className="toolbar__hint">
          Tenez le geste environ une demi-seconde pour déclencher l'action.
        </span>
      </div>
      <div className={`viewer viewer--webcam ${running ? "" : "viewer--idle"}`}>
        <video ref={videoRef} muted playsInline />
        <canvas ref={overlayRef} className="viewer__overlay" />
        {!running && <p className="viewer__placeholder">La webcam est arrêtée</p>}
      </div>
    </div>
  );
}
