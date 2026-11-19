"""Currículo da Fase 6: partidas de TREINO que começam já avançadas.

O problema: o que segura todos os agentes é o fim de jogo (cobra enorme, tabuleiro
quase cheio). Mas, começando sempre do início, uma partida de treino leva centenas
de passos para chegar lá, e o agente vê pouquíssimos fins de jogo. É como treinar
xadrez jogando sempre do zero: você pratica mil aberturas e quase nenhum final.

A solução: um "banco" de situações avançadas tiradas de partidas do agente BFS
(Fase 3) em sementes de TREINO. No reset, com probabilidade `start_prob`, a partida
começa numa dessas situações em vez do início oficial. Daí em diante quem joga é o
agente em treino; o BFS só "montou o tabuleiro".

Regras para não "colar na prova":
- As sementes do banco são sorteadas e nunca coincidem com as 1.000 de avaliação.
- A avaliação (evaluate.py) sempre começa do início oficial.
"""

from __future__ import annotations

import numpy as np

from snake.boards import BOARDS
from snake.evaluation import eval_seeds


def snapshot(game) -> dict:
    """O mínimo para remontar a situação com SnakeGame.load_state(**snapshot)."""
    return {
        "snake": list(game.snake),
        "direction": game.direction,
        "fruits": list(game.fruits),
        "obstacles": list(game.obstacles),
        "fruits_collected": game.fruits_collected,
    }


def collect_start_states(n_games: int, seed: int = 0, min_fruits: int = 20, every: int = 25,
                         board: str = "full") -> tuple[list[dict], list[int]]:
    """Joga `n_games` partidas do BFS e guarda um retrato a cada `every` passos,
    a partir de `min_fruits` frutas. Devolve (retratos, sementes usadas)."""
    from agents.simple import BfsAgent  # import aqui: o motor não depende de agents/
    from snake.env import SnakeGame

    forbidden = set(eval_seeds(1000))
    rng = np.random.default_rng(seed)
    states, seeds = [], []
    while len(seeds) < n_games:
        s = int(rng.integers(0, 2**31 - 1))
        if s in forbidden:
            continue
        seeds.append(s)
        game, agent = SnakeGame(BOARDS[board], seed=s), BfsAgent()
        agent.reset(s)
        while not game.game_over:
            if game.fruits_collected >= min_fruits and game.steps % every == 0 \
                    and len(game.fruits) == game.config.num_fruits:
                states.append(snapshot(game))
            game.step(agent.act(game))
    return states, seeds
