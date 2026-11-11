import { useCallback, useEffect, useRef, useState } from "react";
import { fetchHealth, fetchMetadata } from "./api.js";
import { GestureConfirmer, PROFILES, pickMainHand } from "./gestures.js";
import ImageMode from "./components/ImageMode.jsx";
import WebcamMode from "./components/WebcamMode.jsx";
import StatsPanel from "./components/StatsPanel.jsx";
import ActionPanel from "./components/ActionPanel.jsx";
import DetectionList from "./components/DetectionList.jsx";

const EMPTY_GESTURE = { candidate: null, progress: 0, triggered: null, locked: false };

export default function App() {
  const [mode, setMode] = useState("webcam");
  const [health, setHealth] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [error, setError] = useState(null);
  const [detections, setDetections] = useState([]);
  const [stats, setStats] = useState({});
  const [profileKey, setProfileKey] = useState("presentation");
  const [gesture, setGesture] = useState(EMPTY_GESTURE);
  const [lastAction, setLastAction] = useState(null);

  const confirmer = useRef(new GestureConfirmer());
  const history = useRef([]);

  useEffect(() => {
    fetchHealth()
      .then((h) => {
        setHealth(h);
        if (h.model_loaded) return fetchMetadata().then(setMetadata);
        return null;
      })
      .catch(() => setHealth({ status: "down", model_loaded: false }));
  }, []);

  // Changement de mode ou de profil : on repart de zero (stats et confirmation).
  useEffect(() => {
    confirmer.current.reset();
    history.current = [];
    setGesture(EMPTY_GESTURE);
    setStats({});
  }, [profileKey, mode]);

  const handleResult = useCallback(
    (result) => {
      setError(null);
      setDetections(result.detections);

      const now = performance.now();
      history.current = [...history.current, { now, ms: result.inference_ms }].slice(-30);
      const h = history.current;
      const seconds = (h[h.length - 1].now - h[0].now) / 1000;
      setStats({
        inferenceMs: result.inference_ms,
        roundTripMs: result.roundTripMs,
        fps: h.length > 1 && seconds > 0 ? (h.length - 1) / seconds : null,
        avgInferenceMs: h.reduce((sum, x) => sum + x.ms, 0) / h.length,
      });

      if (mode !== "webcam") return;
      const main = pickMainHand(result.detections);
      const state = confirmer.current.update(main?.label ?? null, now);
      setGesture(state);
      const profile = PROFILES[profileKey];
      if (state.triggered && state.triggered !== profile.lockGesture) {
        const action = profile.actions[state.triggered];
        if (action) setLastAction({ ...action, at: now });
      }
    },
    [mode, profileKey],
  );

  const online = health?.status === "ok";
  const ready = online && health.model_loaded;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand__logo">✋</span>
          <div>
            <h1>TouchFree</h1>
            <p>Contrôle sans contact par gestes de la main</p>
          </div>
        </div>
        <div className={`status ${ready ? "status--ok" : "status--ko"}`}>
          <span className="status__dot" />
          {!health && "Connexion…"}
          {health && !online && "API injoignable"}
          {online && !health.model_loaded && "API en ligne, modèle absent"}
          {ready && `Modèle v${metadata?.version ?? "?"} · ${metadata?.variant ?? ""}`}
        </div>
      </header>

      <nav className="tabs" role="tablist">
        {[
          ["webcam", "Webcam en direct"],
          ["image", "Analyser une image"],
        ].map(([key, label]) => (
          <button
            key={key}
            role="tab"
            aria-selected={mode === key}
            className={`tab ${mode === key ? "tab--active" : ""}`}
            onClick={() => {
              setMode(key);
              setDetections([]);
            }}
          >
            {label}
          </button>
        ))}
      </nav>

      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}

      <main className="layout">
        <div className="layout__main">
          {mode === "webcam" ? (
            <WebcamMode onResult={handleResult} onError={setError} disabled={!ready} />
          ) : (
            <ImageMode onResult={handleResult} onError={setError} disabled={!ready} />
          )}
        </div>
        <aside className="layout__side">
          <StatsPanel stats={stats} />
          {mode === "webcam" && (
            <ActionPanel
              profileKey={profileKey}
              onProfileChange={setProfileKey}
              gesture={gesture}
              lastAction={lastAction}
            />
          )}
          <DetectionList detections={detections} />
        </aside>
      </main>

      <footer className="footer">
        YOLO11 · ONNX Runtime · FastAPI · React — projet Deep Learning &amp; MLOps
      </footer>
    </div>
  );
}
