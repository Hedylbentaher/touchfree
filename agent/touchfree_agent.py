"""Agent local TouchFree : webcam -> API -> touches clavier simulees.

Il tourne sur le PC de l'utilisateur (hors Docker) car il doit acceder a la
webcam et au clavier. Exemple :
    python -m agent.touchfree_agent --profile presentation --show
    python -m agent.touchfree_agent --profile media --dry-run
"""

import argparse
import time
from pathlib import Path

import yaml

from agent.gesture_logic import GestureDebouncer, pick_main_hand

PROFILES_DIR = Path(__file__).parent / "profiles"


def load_profile(name):
    with open(PROFILES_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def press(keys, dry_run):
    if dry_run:
        print(f"  [simulation] touches : {' + '.join(keys)}")
        return
    import pyautogui

    if len(keys) == 1:
        pyautogui.press(keys[0])
    else:
        pyautogui.hotkey(*keys)


def detect(session, api_url, frame, cv2):
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return []
    files = {"file": ("frame.jpg", jpeg.tobytes(), "image/jpeg")}
    response = session.post(f"{api_url}/predict", files=files, timeout=5)
    response.raise_for_status()
    return response.json()["detections"]


def main():
    parser = argparse.ArgumentParser(description="Agent de controle par gestes")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--profile", default="presentation")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--dry-run", action="store_true", help="affiche sans taper")
    parser.add_argument("--show", action="store_true", help="fenetre de controle")
    args = parser.parse_args()

    import cv2
    import requests

    profile = load_profile(args.profile)
    debouncer = GestureDebouncer(
        confirm_frames=profile.get("confirm_frames", 5),
        cooldown_s=profile.get("cooldown_s", 1.0),
        lock_gesture=profile.get("lock_gesture", "stop"),
    )
    actions = profile["actions"]
    camera = cv2.VideoCapture(args.camera)
    session = requests.Session()
    period = 1.0 / args.fps
    print(f"Profil « {profile['name']} » actif. Ctrl+C (ou q) pour quitter.")

    try:
        while True:
            start = time.time()
            ok, frame = camera.read()
            if not ok:
                print("Webcam indisponible.")
                break
            frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            try:
                detections = detect(session, args.api, frame, cv2)
            except requests.RequestException as exc:
                print(f"API injoignable : {exc}")
                time.sleep(1)
                continue

            main_hand = pick_main_hand(detections)
            label = main_hand["label"] if main_hand else None
            decision = debouncer.update(label, time.time())
            if decision.triggered == debouncer.lock_gesture:
                print("Controle verrouille" if decision.locked else "Controle actif")
            elif decision.triggered in actions:
                action = actions[decision.triggered]
                print(f"{decision.triggered} -> {action['description']}")
                press(action["keys"], args.dry_run)

            if args.show:
                text = f"{label or '-'}  {decision.progress:.0%}"
                if decision.locked:
                    text += "  [VERROUILLE]"
                cv2.putText(
                    frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.imshow("TouchFree", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(max(0.0, period - (time.time() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
