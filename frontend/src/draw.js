import { colorOf, labelOf } from "./gestures.js";

// Dessine les boites. `mirrored` : le canvas est retourne en CSS (effet miroir
// de la webcam), on retourne donc le texte pour qu'il reste lisible.
export function drawDetections(ctx, detections, { mirrored = false } = {}) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  const scale = Math.max(width, height) / 640;
  ctx.lineWidth = Math.max(2, 3 * scale);
  ctx.font = `600 ${Math.round(16 * scale)}px system-ui, sans-serif`;

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.box;
    const color = colorOf(det.class_id);
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const text = `${labelOf(det.label)} ${(det.confidence * 100).toFixed(0)} %`;
    const padding = 6 * scale;
    const textWidth = ctx.measureText(text).width + padding * 2;
    const textHeight = 24 * scale;
    const top = Math.max(0, y1 - textHeight);

    ctx.save();
    if (mirrored) {
      ctx.translate(x2, 0);
      ctx.scale(-1, 1);
    } else {
      ctx.translate(x1, 0);
    }
    ctx.fillStyle = color;
    ctx.fillRect(0, top, textWidth, textHeight);
    ctx.fillStyle = "#0b1020";
    ctx.fillText(text, padding, top + textHeight * 0.72);
    ctx.restore();
  }
}
