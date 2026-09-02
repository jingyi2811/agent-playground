import os

import pytest

from crew.board import Board, Task
from crew.roles import classical_verifier
from crew.run import run_crew
from make_video import make
from vision import describe as ds
from vision import scenes as sc


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("clip"))
    truth = make(d, scenes=3, seed=5)
    return d, truth


def test_scene_detection_matches_truth(clip):
    d, truth = clip
    frames, _ = sc.read_frames(os.path.join(d, "video.mp4"))
    scenes = sc.detect_scenes(frames)
    assert [(s.start_frame, s.end_frame) for s in scenes] == \
        [(s["start_frame"], s["end_frame"]) for s in truth["scenes"]]


def test_classical_description_finds_objects(clip):
    d, truth = clip
    frames, _ = sc.read_frames(os.path.join(d, "video.mp4"))
    s = truth["scenes"][0]
    cap = ds.classical(frames, s["start_frame"], s["end_frame"])
    assert cap["background"] == s["background"]
    got = {(o["shape"], o["color"]) for o in cap["objects"]}
    want = {(o["shape"], o["color"]) for o in s["objects"]}
    assert want <= got


def test_board_claim_is_exclusive(tmp_path):
    b = Board(str(tmp_path))
    b.post(Task(id="t1", kind="describe", payload={}), by="test")
    assert b.claim("t1", "a") is not None
    assert b.claim("t1", "b") is None
    b.finish("t1", "a", {"x": 1})
    assert b.get("t1").status == "done"


def test_dependencies_gate_readiness(tmp_path):
    b = Board(str(tmp_path))
    b.post(Task(id="d", kind="describe", payload={}), by="t")
    b.post(Task(id="v", kind="verify", payload={}, depends_on=["d"]), by="t")
    assert [t.id for t in b.ready()] == ["d"]
    b.claim("d", "w"); b.finish("d", "w", {})
    assert [t.id for t in b.ready()] == ["v"]


def test_verifier_rejects_planted_error(clip):
    d, truth = clip
    frames, _ = sc.read_frames(os.path.join(d, "video.mp4"))
    s = truth["scenes"][0]
    claims = ds.claims_from(ds.classical(frames, s["start_frame"], s["end_frame"]))
    claims.append("there is a purple hexagon")
    verdicts = classical_verifier(frames, s["start_frame"], s["end_frame"], claims)
    assert verdicts[-1] is False and all(verdicts[:-1])


def test_full_crew_run(clip, tmp_path):
    d, truth = clip
    r = run_crew(os.path.join(d, "video.mp4"), str(tmp_path / "run"), workers=2)
    assert r["board"] == {"done": 2 * len(truth["scenes"]) + 1}
    assert len(r["summary"]["scenes"]) == len(truth["scenes"])
    assert os.path.exists(r["poster"]["path"])
