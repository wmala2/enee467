"""The publish gate must refuse bad policies, which is the only behaviour worth testing.

rover_control/rl_rover.py downloads whatever is in the Hub repo and drives the motors with
it, so every check here stands between a mistake and the hardware. A gate that only ever
passes is decoration; these assert each rejection fires.
"""

import json
from pathlib import Path
import sys

from envs import contract
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish_policy  # ty: ignore[unresolved-import]


class _Space:
    def __init__(self, shape):
        self.shape = shape


class _Model:
    def __init__(self, obs_shape, action_shape):
        self.observation_space = _Space(obs_shape)
        self.action_space = _Space(action_shape)


def test_contract_accepts_the_declared_shapes():
    publish_policy.check_contract(_Model((contract.OBS_LEN,), (contract.ACTION_LEN,)))


def test_contract_rejects_an_image_observation():
    """The exact mismatch that exists today: the older policy was image-based at 64x64x1."""
    with pytest.raises(SystemExit, match="observation is"):
        publish_policy.check_contract(_Model((64, 64, 1), (contract.ACTION_LEN,)))


def test_contract_rejects_a_wrong_action_width():
    with pytest.raises(SystemExit, match="action is"):
        publish_policy.check_contract(_Model((contract.OBS_LEN,), (4,)))


def _write_qualification(run, rates, episodes=40, sha="abc"):
    directory = run / "evaluation-40"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "qualification.json").write_text(
        json.dumps({"rates": rates, "episodes_per_track": episodes, "model_sha256": sha})
    )
    return directory / "qualification.json"


def test_publishing_without_evidence_is_refused(tmp_path):
    with pytest.raises(SystemExit, match=r"qualification\.json"):
        publish_policy.load_qualification(tmp_path, 0.9)


def test_rates_below_the_requirement_are_refused(tmp_path):
    _write_qualification(tmp_path, {"circle": 1.0, "figure8": 0.75, "oval": 1.0})
    with pytest.raises(SystemExit, match="below"):
        publish_policy.load_qualification(tmp_path, 0.9)


def test_the_record_covering_the_most_episodes_wins(tmp_path):
    """Two records for one checkpoint must resolve to the better-sampled one.

    A perfect 20/20 bounds the true rate at only 86.1%, so a run carrying both a 20- and a
    40-episode record must publish against the stronger evidence rather than whichever sorts
    first.
    """
    small = tmp_path / "evaluation"
    small.mkdir()
    (small / "qualification.json").write_text(
        json.dumps({"rates": {"circle": 1.0}, "episodes_per_track": 20, "model_sha256": "abc"})
    )
    _write_qualification(tmp_path, {"circle": 1.0}, episodes=40)
    path, qualification = publish_policy.load_qualification(tmp_path, 0.9)
    assert qualification["episodes_per_track"] == 40, f"picked {path}"
