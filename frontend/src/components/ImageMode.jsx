import { useEffect, useRef, useState } from "react";
import { predict } from "../api.js";
import { drawDetections } from "../draw.js";

export default function ImageMode({ onResult, onError, disabled }) {
  const [imageUrl, setImageUrl] = useState(null);
  const [detections, setDetections] = useState([]);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => () => imageUrl && URL.revokeObjectURL(imageUrl), [imageUrl]);

  // Redessine les boites quand l'image ou les detections changent.
  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !img.naturalWidth) return;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    drawDetections(canvas.getContext("2d"), detections);
  }, [detections, imageUrl]);

  async function analyse(file) {
    if (!file || !file.type.startsWith("image/")) {
      onError("Choisissez une image JPEG ou PNG.");
      return;
    }
    setImageUrl(URL.createObjectURL(file));
    setDetections([]);
    setBusy(true);
    try {
      const result = await predict(file, file.name);
      setDetections(result.detections);
      onResult(result);
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function onDrop(event) {
    event.preventDefault();
    setDragging(false);
    analyse(event.dataTransfer.files[0]);
  }

  return (
    <div className="mode">
      <label
        className={`dropzone ${dragging ? "dropzone--active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          type="file"
          accept="image/jpeg,image/png"
          disabled={disabled || busy}
          onChange={(e) => analyse(e.target.files[0])}
        />
        <span className="dropzone__title">
          {busy ? "Analyse en cours…" : "Déposez une photo de main ici"}
        </span>
        <span className="dropzone__hint">ou cliquez pour choisir un fichier</span>
      </label>

      {imageUrl && (
        <div className="viewer">
          <img
            ref={imgRef}
            src={imageUrl}
            alt="Image analysée"
            onLoad={() => setDetections((d) => [...d])}
          />
          <canvas ref={canvasRef} className="viewer__overlay" />
        </div>
      )}
    </div>
  );
}
