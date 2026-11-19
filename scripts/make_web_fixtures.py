"""Gera os "gabaritos" do teste de paridade Python x JavaScript (Fase 8).

A demo web reescreve o motor do jogo e os 18 números da observação em JS. Para ter
certeza de que a tradução está fiel, o Python joga partidas reais, tira "fotos" de
estados e anota o que o JS precisa responder em cada um:

- os 18 números da observação (o que a rede vê);
- as direções seguras;
- as células onde um obstáculo poderia nascer;
- o que acontece ao dar um passo em cada direção (morre? come? onde fica o corpo?).

O teste web/tests/parity.test.js lê este arquivo e compara. É como um teste de
"snapshot" do Jest: o Python é a referência.

Como rodar:
    .venv\\Scripts\\python.exe scripts\\make_web_fixtures.py
    node --test web/tests/parity.test.js
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from agents.ppo import PPOAgent  # noqa: E402
from snake.gym_env import space_feature_observation  # noqa: E402
from snake.env import SnakeGame  # noqa: E402

OUT = ROOT / "web" / "tests" / "fixtures.json"


def snapshot(game: SnakeGame) -> dict:
    state = {
        "snake": [list(c) for c in game.snake],
        "direction": game.direction,
        "fruits": [list(c) for c in game.fruits],
        "obstacles": [list(c) for c in game.obstacles],
        "stepsSinceFruit": game.steps_since_fruit,
    }
    steps = []
    for action in range(4):
        trial = SnakeGame(seed=0)
        trial.load_state(game.snake, game.direction, game.fruits, game.obstacles, game.fruits_collected)
        trial.steps_since_fruit = game.steps_since_fruit
        result = trial.step(action)
        # Só o que não depende de sorteio (onde nasce fruta/obstáculo depende do gerador).
        steps.append({"endReason": result.end_reason if result.end_reason != "win" else "win",
                      "ateFruit": result.ate_fruit,
                      "snake": [list(c) for c in trial.snake]})
    return {
        "state": state,
        "features": [float(x) for x in space_feature_observation(game)],
        "safe": game.safe_directions(),
        "eligibleObstacles": [list(c) for c in game.eligible_obstacle_cells()],
        "steps": steps,
    }


def main() -> None:
    agent = PPOAgent.load(ROOT / "experiments" / "fase6_ppo" / "model.zip")
    rng = np.random.default_rng(0)
    cases = []
    for seed in range(40):
        game = SnakeGame(seed=10_000 + seed)
        while not game.game_over:
            # Fotos espalhadas pela partida (começo, meio e fim de jogo lotado).
            if rng.random() < 0.02:
                cases.append(snapshot(game))
            # 10% das jogadas sorteadas (entre as seguras), para cair em estados que a rede não visitaria.
            safe = game.safe_directions()
            action = int(rng.choice(safe)) if safe and rng.random() < 0.1 else agent.act(game)
            game.step(action)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    lengths = [len(c["state"]["snake"]) for c in cases]
    print(f"{len(cases)} estados salvos em {OUT} (cobra de {min(lengths)} a {max(lengths)} células)")


if __name__ == "__main__":
    main()
