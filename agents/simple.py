"""Agentes sem ML (Fase 3): aleatório, aleatório seguro, guloso e BFS + flood fill.

Todos seguem a mesma "interface" (o mesmo formato), para o evaluate.py tratá-los
igual — e, depois, os agentes de RL também:

    agent.reset(seed)    # começo de partida (sorteios do agente ficam reproduzíveis)
    agent.act(game)      # devolve a direção: 0 cima, 1 direita, 2 baixo, 3 esquerda

Em JavaScript, seria como várias classes com os mesmos métodos `reset` e `act`
(o famoso "duck typing": se anda como pato e grasna como pato, é pato).

Regra de ouro (seção 9 da especificação): o agente só CONSULTA o jogo, por
`game.get_state()` e `game.safe_directions()`. Ele nunca chama `game.step()`
nem olha o gerador aleatório do jogo (isso seria espiar onde a próxima
fruta/obstáculo vai nascer).
"""

from __future__ import annotations

import numpy as np

from agents.grid import (
    body_after_path,
    bfs_paths_to_fruits,
    direction_between,
    flood_fill,
    tail_reachable,
)
from snake.env import DIRECTION_DELTAS, DOWN, LEFT, RIGHT, UP, SnakeGame

ALL_DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


class RandomAgent:
    """Sorteia qualquer uma das 4 direções. É o "piso": qualquer agente tem que bater ele."""

    name = "random"

    def __init__(self):
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, game: SnakeGame) -> int:
        return int(self.rng.choice(ALL_DIRECTIONS))


class SafeRandomAgent(RandomAgent):
    """Sorteia só entre as direções que não matam no passo seguinte.

    `class SafeRandomAgent(RandomAgent)` = herança, como `extends` no JS: ele
    reaproveita o __init__ e o reset do pai e troca só o act.
    """

    name = "safe_random"

    def act(self, game: SnakeGame) -> int:
        options = game.safe_directions() or [game.get_state()["snake_direction"]]
        return int(self.rng.choice(options))


class GreedyAgent:
    """Guloso: entre as direções seguras, vai para a que mais aproxima de alguma fruta.

    "Aproximar" = distância de Manhattan (|Δlinha| + |Δcoluna|), que ignora
    paredes e obstáculos no caminho. É rápido e esperto no curto prazo, mas não
    enxerga armadilhas: entra em corredores sem saída se a fruta estiver lá.
    Não sorteia nada: com a mesma partida, faz sempre o mesmo.
    """

    name = "greedy"

    def reset(self, seed: int | None = None) -> None:
        pass  # nada para reiniciar

    def act(self, game: SnakeGame) -> int:
        state = game.get_state()
        current = state["snake_direction"]
        safe = game.safe_directions()
        if not safe:
            return current  # vai morrer de qualquer jeito
        head = state["snake_head"]

        def distance_after(direction: int) -> int:
            dr, dc = DIRECTION_DELTAS[direction]
            r, c = head[0] + dr, head[1] + dc
            return min(abs(r - fr) + abs(c - fc) for fr, fc in state["fruits"])

        # Empate: prefere seguir reto (direção atual), depois a ordem cima/direita/baixo/esquerda.
        # min() com uma tupla como chave compara item a item, como ordenar por
        # vários critérios no JS: (a, b) => a.dist - b.dist || a.reto - b.reto.
        return min(safe, key=lambda d: (distance_after(d), d != current, d))


class BfsAgent:
    """BFS + flood fill: o menor caminho até uma fruta, mas só se for seguro.

    A cada passo:
    1. Calcula o menor caminho real (desviando de corpo e obstáculos) até cada fruta.
    2. Do mais curto ao mais longo: imagina a cobra depois de comer aquela fruta.
       Se de lá ela ainda alcança a própria cauda, o plano é seguro -> dá o 1º passo.
    3. Se nenhum plano é seguro ("modo sobrevivência"): entre as direções que não
       matam já, escolhe a que mantém a cauda alcançável e deixa mais espaço livre.

    Limitação conhecida: o agente não prevê onde nascerão a próxima fruta e o
    próximo obstáculo (são sorteados). Um obstáculo novo pode fechar o caminho
    até a cauda depois que ele já se comprometeu.
    """

    name = "bfs"

    def reset(self, seed: int | None = None) -> None:
        pass  # determinístico

    def act(self, game: SnakeGame) -> int:
        state = game.get_state()
        body = state["snake_body"]
        obstacles = set(state["obstacles"])
        height, width = state["board_height"], state["board_width"]

        # Passos 1 e 2: o caminho mais curto que não leva a uma armadilha.
        for path in bfs_paths_to_fruits(state):
            after = body_after_path(body, path, ate_at_end=True)
            if tail_reachable(after, obstacles, height, width):
                return direction_between(body[0], path[0])

        # Passo 3: modo sobrevivência.
        return self.survival_move(game, state)

    def survival_move(self, game: SnakeGame, state: dict) -> int:
        body = state["snake_body"]
        obstacles = set(state["obstacles"])
        fruits = set(state["fruits"])
        height, width = state["board_height"], state["board_width"]
        current = state["snake_direction"]
        head = body[0]

        best, best_score = current, None
        for direction in game.safe_directions():
            dr, dc = DIRECTION_DELTAS[direction]
            cell = (head[0] + dr, head[1] + dc)
            after = body_after_path(body, [cell], ate_at_end=cell in fruits)
            space = flood_fill(cell, obstacles | set(after), height, width)
            # Critérios, em ordem: cauda alcançável > mais espaço > seguir reto.
            score = (tail_reachable(after, obstacles, height, width), space, direction == current)
            if best_score is None or score > best_score:
                best, best_score = direction, score
        return best


# Registro de agentes por nome (um "objeto de fábricas" do JS): o evaluate.py
# recebe o nome na linha de comando e cria o agente com AGENTS[nome]().
AGENTS = {
    RandomAgent.name: RandomAgent,
    SafeRandomAgent.name: SafeRandomAgent,
    GreedyAgent.name: GreedyAgent,
    BfsAgent.name: BfsAgent,
}
