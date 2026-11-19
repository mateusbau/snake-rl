"""Testes da Fase 8: a demo web joga igual ao Python?

1. O ONNX exportado escolhe a mesma ação que o modelo do SB3 em partidas reais.
2. O motor e a observação em JS batem com o Python (roda o teste do Node, se o Node existir).
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from agents.ppo import PPOAgent
from snake.env import SnakeGame
from snake.gym_env import space_feature_observation

ROOT = Path(__file__).resolve().parent.parent
ONNX = ROOT / "web" / "models" / "ppo_final.onnx"
MODEL = ROOT / "experiments" / "fase6_ppo" / "model.zip"


@pytest.mark.skipif(not ONNX.exists() or not MODEL.exists(), reason="rode scripts/export_onnx.py antes")
def test_onnx_plays_like_the_sb3_model():
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(ONNX))
    agent = PPOAgent.load(MODEL)
    for seed in range(3):
        game = SnakeGame(seed=seed)
        while not game.game_over:
            obs = space_feature_observation(game)[None, :]   # [None, :] = lote de 1, formato (1, 18)
            onnx_action = int(np.argmax(session.run(["logits"], {"obs": obs})[0]))
            sb3_action = agent.act(game)
            assert onnx_action == sb3_action
            game.step(sb3_action)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js não instalado")
def test_javascript_engine_matches_python():
    result = subprocess.run(["node", "--test", "web/tests/parity.test.js"], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
