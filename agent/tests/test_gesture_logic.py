"""Tests de la logique anti faux declenchements de l'agent."""

from agent.gesture_logic import GestureDebouncer, pick_main_hand


def feed(debouncer, labels, start=0.0, step=0.1):
    return [debouncer.update(lab, start + i * step) for i, lab in enumerate(labels)]


def test_la_plus_grande_main_est_choisie():
    small = {"label": "like", "box": [0, 0, 10, 10]}
    big = {"label": "palm", "box": [0, 0, 100, 100]}
    assert pick_main_hand([small, big])["label"] == "palm"
    assert pick_main_hand([]) is None


def test_un_geste_se_declenche_apres_5_images_et_une_seule_fois():
    d = GestureDebouncer(confirm_frames=5)
    triggered = [x.triggered for x in feed(d, ["like"] * 12)]
    assert triggered.count("like") == 1
    assert triggered[4] == "like"


def test_un_geste_interrompu_ne_declenche_rien():
    d = GestureDebouncer(confirm_frames=5)
    labels = ["like"] * 4 + ["no_gesture"] + ["like"] * 4
    assert all(x.triggered is None for x in feed(d, labels))


def test_la_pause_empeche_deux_actions_trop_rapprochees():
    d = GestureDebouncer(confirm_frames=2, cooldown_s=1.0)
    out = feed(d, ["like", "like", "dislike", "dislike"], step=0.1)
    assert [x.triggered for x in out] == [None, "like", None, None]


def test_le_verrou_bloque_les_actions_puis_les_libere():
    d = GestureDebouncer(confirm_frames=2, cooldown_s=0.0, lock_gesture="stop")
    out = feed(d, ["stop", "stop", "like", "like", "stop", "stop", "like", "like"])
    assert out[1].locked is True
    assert out[3].triggered is None  # verrouille : pas d'action
    assert out[5].locked is False
    assert out[7].triggered == "like"
