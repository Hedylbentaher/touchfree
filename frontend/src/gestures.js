// Noms lisibles des 19 classes et profils « geste -> action ».
// Meme logique que l'agent local (agent/gesture_logic.py).

export const GESTURE_LABELS = {
  call: "Appel 🤙",
  dislike: "Pouce bas 👎",
  fist: "Poing ✊",
  four: "Quatre",
  like: "Pouce haut 👍",
  mute: "Silence 🤫",
  ok: "OK 👌",
  one: "Un ☝️",
  palm: "Paume ✋",
  peace: "Paix ✌️",
  peace_inverted: "Paix inversée",
  rock: "Rock 🤘",
  stop: "Stop 🖐",
  stop_inverted: "Stop inversé",
  three: "Trois",
  three2: "Trois (variante)",
  two_up: "Deux doigts levés",
  two_up_inverted: "Deux doigts baissés",
  no_gesture: "Main au repos",
};

export const labelOf = (name) => GESTURE_LABELS[name] ?? name;

export const PROFILES = {
  presentation: {
    name: "Présentation",
    lockGesture: "stop",
    actions: {
      like: { keys: "→", description: "Diapositive suivante" },
      dislike: { keys: "←", description: "Diapositive précédente" },
      two_up: { keys: "Ctrl +", description: "Zoom avant" },
      two_up_inverted: { keys: "Ctrl −", description: "Zoom arrière" },
      fist: { keys: "Échap", description: "Quitter le diaporama" },
    },
  },
  media: {
    name: "Lecteur vidéo",
    lockGesture: "stop",
    actions: {
      palm: { keys: "Espace", description: "Lecture / pause" },
      like: { keys: "Volume +", description: "Monter le son" },
      dislike: { keys: "Volume −", description: "Baisser le son" },
      mute: { keys: "Muet", description: "Couper le son" },
      one: { keys: "→", description: "Avancer de 10 s" },
    },
  },
};

// Couleur stable par classe (teinte repartie sur le cercle chromatique).
export function colorOf(classId) {
  return `hsl(${(classId * 47) % 360} 85% 55%)`;
}

// La main principale = la plus grande boite (la plus proche de la camera).
export function pickMainHand(detections) {
  let best = null;
  for (const det of detections) {
    const [x1, y1, x2, y2] = det.box;
    const area = (x2 - x1) * (y2 - y1);
    if (!best || area > best.area) best = { ...det, area };
  }
  return best;
}

// Un geste ne declenche une action qu'apres N images consecutives identiques,
// puis une pause (cooldown) evite les doubles declenchements.
export class GestureConfirmer {
  constructor({ confirmFrames = 5, cooldownMs = 1000, lockGesture = "stop" } = {}) {
    this.confirmFrames = confirmFrames;
    this.cooldownMs = cooldownMs;
    this.lockGesture = lockGesture;
    this.reset();
  }

  reset() {
    this.candidate = null;
    this.count = 0;
    this.lastTrigger = -Infinity;
    this.locked = false;
  }

  update(label, now = performance.now()) {
    if (!label || label === "no_gesture") {
      this.candidate = null;
      this.count = 0;
      return this.state(null);
    }
    this.count = label === this.candidate ? this.count + 1 : 1;
    this.candidate = label;

    let triggered = null;
    const ready = this.count === this.confirmFrames;
    const cooled = now - this.lastTrigger >= this.cooldownMs;
    if (ready && cooled) {
      if (label === this.lockGesture) {
        this.locked = !this.locked;
        triggered = label;
      } else if (!this.locked) {
        triggered = label;
      }
      if (triggered) this.lastTrigger = now;
    }
    return this.state(triggered);
  }

  state(triggered) {
    return {
      candidate: this.candidate,
      progress: Math.min(this.count / this.confirmFrames, 1),
      triggered,
      locked: this.locked,
    };
  }
}
