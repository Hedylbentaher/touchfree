import { PROFILES, labelOf } from "../gestures.js";

export default function ActionPanel({ profileKey, onProfileChange, gesture, lastAction }) {
  const profile = PROFILES[profileKey];
  return (
    <section className="card">
      <div className="card__header">
        <h2>Contrôle par gestes</h2>
        <select value={profileKey} onChange={(e) => onProfileChange(e.target.value)}>
          {Object.entries(PROFILES).map(([key, p]) => (
            <option key={key} value={key}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className={`gesture ${gesture.locked ? "gesture--locked" : ""}`}>
        <span className="gesture__name">
          {gesture.candidate ? labelOf(gesture.candidate) : "Aucun geste"}
        </span>
        <div className="progress" aria-label="Confirmation du geste">
          <div className="progress__bar" style={{ width: `${gesture.progress * 100}%` }} />
        </div>
        <span className="gesture__status">
          {gesture.locked
            ? "Verrouillé : faites « Stop » pour reprendre"
            : "Confirmation sur 5 images consécutives"}
        </span>
      </div>

      {lastAction && (
        <p className="last-action" key={lastAction.at}>
          ⚡ {lastAction.description} <kbd>{lastAction.keys}</kbd>
        </p>
      )}

      <ul className="mapping">
        {Object.entries(profile.actions).map(([name, action]) => (
          <li key={name} className={gesture.candidate === name ? "mapping--active" : ""}>
            <span>{labelOf(name)}</span>
            <span>{action.description}</span>
            <kbd>{action.keys}</kbd>
          </li>
        ))}
        <li className={gesture.candidate === profile.lockGesture ? "mapping--active" : ""}>
          <span>{labelOf(profile.lockGesture)}</span>
          <span>Verrouiller / déverrouiller</span>
          <kbd>🔒</kbd>
        </li>
      </ul>
    </section>
  );
}
