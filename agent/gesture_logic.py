"""Logique de decision de l'agent, sans webcam ni clavier : facile a tester.

Regles anti faux declenchements :
  1. on ne regarde que la main la plus grande (la plus proche) ;
  2. un geste doit etre vu sur N images consecutives (~0,5 s a 10 img/s) ;
  3. apres une action, pause (cooldown) avant la suivante ;
  4. le geste de verrouillage active ou coupe le controle.
"""

from dataclasses import dataclass

IGNORED = {None, "no_gesture"}


def pick_main_hand(detections):
    """La plus grande boite = la main la plus proche de la camera."""

    def area(det):
        x1, y1, x2, y2 = det["box"]
        return (x2 - x1) * (y2 - y1)

    return max(detections, key=area) if detections else None


@dataclass
class Decision:
    candidate: str | None
    progress: float
    triggered: str | None
    locked: bool


class GestureDebouncer:
    def __init__(self, confirm_frames=5, cooldown_s=1.0, lock_gesture="stop"):
        self.confirm_frames = confirm_frames
        self.cooldown_s = cooldown_s
        self.lock_gesture = lock_gesture
        self.candidate = None
        self.count = 0
        self.last_trigger = float("-inf")
        self.locked = False

    def update(self, label, now):
        if label in IGNORED:
            self.candidate, self.count = None, 0
            return self._decision(None)

        self.count = self.count + 1 if label == self.candidate else 1
        self.candidate = label

        triggered = None
        ready = self.count == self.confirm_frames  # une seule fois par geste tenu
        cooled = now - self.last_trigger >= self.cooldown_s
        if ready and cooled:
            if label == self.lock_gesture:
                self.locked = not self.locked
                triggered = label
            elif not self.locked:
                triggered = label
            if triggered:
                self.last_trigger = now
        return self._decision(triggered)

    def _decision(self, triggered):
        progress = min(self.count / self.confirm_frames, 1.0)
        return Decision(self.candidate, progress, triggered, self.locked)
