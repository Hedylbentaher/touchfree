function Stat({ label, value, unit, hint }) {
  return (
    <div className="stat" title={hint}>
      <span className="stat__label">{label}</span>
      <span className="stat__value">
        {value ?? "–"}
        {value != null && <small>{unit}</small>}
      </span>
    </div>
  );
}

export default function StatsPanel({ stats }) {
  const fmt = (v) => (v == null ? null : v.toFixed(v < 10 ? 1 : 0));
  return (
    <section className="card">
      <h2>Performances</h2>
      <div className="stats">
        <Stat
          label="Inférence serveur"
          value={fmt(stats.inferenceMs)}
          unit=" ms"
          hint="Pré-traitement + modèle ONNX + NMS, mesuré par l'API"
        />
        <Stat
          label="Aller-retour"
          value={fmt(stats.roundTripMs)}
          unit=" ms"
          hint="Envoi de l'image + inférence + réponse, mesuré par le navigateur"
        />
        <Stat label="Images / s" value={fmt(stats.fps)} unit="" hint="Webcam uniquement" />
        <Stat
          label="Moyenne inférence"
          value={fmt(stats.avgInferenceMs)}
          unit=" ms"
          hint="Moyenne glissante sur les 30 dernières requêtes"
        />
      </div>
    </section>
  );
}
