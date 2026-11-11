"""Tests unitaires du pre-traitement, du decodage YOLO et de la NMS."""

import numpy as np
import pytest

from app.inference import OnnxDetector, decode, letterbox, nms

NB_CLASSES = 19
ANCHORS = 5376  # (512/8)^2 + (512/16)^2 + (512/32)^2


def fake_output(cx=256, cy=256, w=100, h=150, class_id=8, score=0.9):
    out = np.zeros((1, 4 + NB_CLASSES, ANCHORS), dtype=np.float32)
    out[0, :4, 0] = [cx, cy, w, h]
    out[0, 4 + class_id, 0] = score
    return out


def test_letterbox_garde_les_proportions_et_centre():
    image = np.zeros((768, 1024, 3), dtype=np.uint8)  # H=768, W=1024
    square, ratio, (dx, dy) = letterbox(image, 512)
    assert square.shape == (512, 512, 3)
    assert ratio == pytest.approx(0.5)
    assert (dx, dy) == (0, 64)  # bandes grises en haut et en bas
    assert square[0, 0, 0] == 114


def test_nms_supprime_les_doublons_mais_garde_les_mains_separees():
    boxes = np.array(
        [[0, 0, 100, 100], [5, 5, 105, 105], [300, 300, 400, 400]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.5) == [0, 2]


def test_decode_revient_aux_pixels_de_l_image_d_origine():
    # Image 1024x768 -> ratio 0.5, decalage vertical 64 px.
    dets = decode(fake_output(), 0.35, 0.5, 0.5, (0, 64), (1024, 768))
    assert len(dets) == 1
    assert dets[0]["class_id"] == 8
    assert dets[0]["box"] == pytest.approx([412, 234, 612, 534], abs=0.5)


def test_decode_ignore_les_scores_trop_faibles():
    assert decode(fake_output(score=0.1), 0.35, 0.5, 1.0, (0, 0), (512, 512)) == []


def test_detecteur_onnx_de_bout_en_bout(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    constant = numpy_helper.from_array(fake_output(), name="const_out")
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Constant", [], ["const"], value=constant),
            helper.make_node("Identity", ["const"], ["output0"]),
        ],
        name="faux_yolo",
        inputs=[
            helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 512, 512])
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output0", TensorProto.FLOAT, [1, 4 + NB_CLASSES, ANCHORS]
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "faux.onnx"
    onnx.save(model, str(path))

    names = [f"classe_{i}" for i in range(NB_CLASSES)]
    detector = OnnxDetector(path, names, imgsz=512)
    dets, elapsed_ms = detector.predict(np.zeros((768, 1024, 3), dtype=np.uint8))
    assert dets[0]["label"] == "classe_8"
    assert elapsed_ms > 0
