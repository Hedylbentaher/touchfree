import { colorOf, labelOf } from "../gestures.js";

export default function DetectionList({ detections }) {
  return (
    <section className="card">
      <h2>Détections</h2>
      {detections.length === 0 ? (
        <p className="muted">Aucune main détectée pour l'instant.</p>
      ) : (
        <ul className="detections">
          {detections.map((det, i) => (
            <li key={i}>
              <span className="dot" style={{ background: colorOf(det.class_id) }} />
              <span className="detections__label">{labelOf(det.label)}</span>
              <span className="detections__conf">{(det.confidence * 100).toFixed(1)} %</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
