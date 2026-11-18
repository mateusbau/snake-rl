"""Avaliação padronizada de agentes: as MESMAS 1.000 sementes para todo mundo.

Por que sementes fixas? A sorte pesa muito no Snake com obstáculos (onde nasce
cada fruta e cada obstáculo). Se o agente A jogasse partidas "fáceis" e o B
partidas "difíceis", a comparação seria injusta. Com as mesmas sementes, todos
enfrentam exatamente os mesmos sorteios — como uma prova igual para a turma toda.

Qualquer "agente" serve, desde que tenha `reset(seed)` e `act(game)`.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from snake.env import GameConfig, SnakeGame

# Semente "mestra" das sementes de avaliação. NUNCA mude: mudar = trocar a prova.
EVAL_MASTER_SEED = 20261118
EVAL_GAMES = 1000


def eval_seeds(n: int = EVAL_GAMES) -> list[int]:
    """As sementes oficiais de avaliação (sempre as mesmas).

    As n primeiras são sempre as mesmas para qualquer n: eval_seeds(10) é o
    começo de eval_seeds(1000). Assim um teste rápido usa um pedaço da prova oficial.
    """
    rng = np.random.default_rng(EVAL_MASTER_SEED)
    # .tolist() converte de números do numpy para int do Python (o JSON agradece).
    return rng.integers(0, 2**31 - 1, size=max(n, EVAL_GAMES)).tolist()[:n]


def play_game(agent, seed: int, config: GameConfig | None = None) -> dict:
    """Joga uma partida inteira com a semente dada e devolve o resultado + as ações."""
    game = SnakeGame(config, seed=seed)
    agent.reset(seed)  # a mesma semente para os sorteios do próprio agente
    actions = []
    while not game.game_over:
        action = agent.act(game)
        actions.append(action)
        game.step(action)
    cells = game.config.height * game.config.width
    return {
        "seed": seed,
        "fruits": game.fruits_collected,
        "steps": game.steps,
        "obstacles": game.obstacles_generated,
        "length": len(game.snake),
        "board_filled": round(len(game.snake) / cells, 4),
        "end_reason": game.end_reason,
        "actions": actions,
    }


def evaluate(agent, seeds: list[int], config: GameConfig | None = None, progress: bool = False) -> list[dict]:
    """Joga uma partida por semente. Devolve a lista de resultados (na ordem das sementes)."""
    results = []
    start = time.perf_counter()
    for i, seed in enumerate(seeds, 1):
        results.append(play_game(agent, seed, config))
        if progress and (i % 100 == 0 or i == len(seeds)):
            print(f"  {agent.name}: {i}/{len(seeds)} partidas ({time.perf_counter() - start:.0f} s)", flush=True)
    return results


def summarize(results: list[dict]) -> dict:
    """As métricas do CLAUDE.md para um conjunto de partidas."""
    fruits = np.array([r["fruits"] for r in results])
    filled = np.array([r["board_filled"] for r in results])
    steps = np.array([r["steps"] for r in results])
    reasons = Counter(r["end_reason"] for r in results)
    n = len(results)
    return {
        "games": n,
        "fruits_mean": round(float(fruits.mean()), 2),
        "fruits_median": float(np.median(fruits)),
        "fruits_best": int(fruits.max()),
        "fruits_std": round(float(fruits.std()), 2),
        # Erro padrão: o quanto a média "balança" de amostra para amostra (fase-2.md).
        "fruits_sem": round(float(fruits.std() / np.sqrt(n)), 2),
        "board_filled_mean": round(float(filled.mean()), 4),
        "board_filled_max": round(float(filled.max()), 4),
        "steps_mean": round(float(steps.mean()), 1),
        "win_rate": reasons.get("win", 0) / n,
        "end_reasons": {k: v / n for k, v in reasons.most_common()},
    }
