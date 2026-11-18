"""Q-learning com tabela (Fase 4): o RL mais simples que existe.

A ideia em uma frase: guardar, para cada situação (estado) e cada ação, uma nota
Q(estado, ação) = "quanta recompensa espero juntar daqui para frente se fizer
essa ação agora". Jogar = escolher a ação de maior nota. Aprender = corrigir a
nota com o que de fato aconteceu:

    Q[s][a] += alpha * (recompensa + gamma * max(Q[s']) - Q[s][a])
                        \\_____________ alvo _____________/

- alpha (taxa de aprendizado): quanto da correção aplicar (0,1 = 10%).
- gamma (desconto): quanto o futuro vale em relação ao agora (0,9).
- s' = estado seguinte. Se a partida acabou de verdade, o futuro vale 0.

A "tabela" é um dicionário: chave = estado (os bytes da observação), valor = 4
notas (uma por direção). Em JS seria um Map<string, number[]>.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from snake.env import SnakeGame
from snake.gym_env import OBSERVERS


class QLearningAgent:
    name = "qlearning"

    def __init__(self, obs_type: str = "grid", alpha: float = 0.1, gamma: float = 0.9, seed: int | None = None):
        self.obs_type = obs_type
        self.observe = OBSERVERS[obs_type]
        self.alpha = alpha
        self.gamma = gamma
        self.q: dict[bytes, np.ndarray] = {}
        self.rng = np.random.default_rng(seed)
        # Contadores da avaliação: em quantos passos o agente caiu num estado que
        # nunca viu no treino (e por isso teve que chutar).
        self.eval_steps = 0
        self.eval_unseen = 0

    # ---- tabela ----
    @staticmethod
    def key(obs: np.ndarray) -> bytes:
        """Transforma a observação numa chave de dicionário (arrays não podem ser chave)."""
        return obs.tobytes()

    def choose(self, obs: np.ndarray, epsilon: float) -> tuple[int, bool]:
        """Escolhe a ação. Devolve (ação, estado_era_conhecido).

        epsilon-greedy: com probabilidade epsilon, EXPLORA (ação sorteada); senão,
        APROVEITA (melhor nota). Empates e estados nunca vistos -> sorteio.
        """
        values = self.q.get(self.key(obs))
        if values is None:
            return int(self.rng.integers(4)), False
        if self.rng.random() < epsilon:
            return int(self.rng.integers(4)), True
        best = np.flatnonzero(values == values.max())  # todas as ações empatadas no topo
        return int(self.rng.choice(best)), True

    def learn(self, obs, action: int, reward: float, next_obs, terminated: bool) -> float:
        """Uma atualização de Q-learning. Devolve o erro (alvo - nota antiga)."""
        values = self.q.setdefault(self.key(obs), np.zeros(4))
        future = 0.0
        if not terminated:
            next_values = self.q.get(self.key(next_obs))
            future = 0.0 if next_values is None else float(next_values.max())
        target = reward + self.gamma * future
        error = target - values[action]
        values[action] += self.alpha * error
        return float(error)

    # ---- interface de agente (igual aos da Fase 3, para o evaluate) ----
    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, game: SnakeGame) -> int:
        action, seen = self.choose(self.observe(game), epsilon=0.0)  # avaliação: sem explorar
        self.eval_steps += 1
        self.eval_unseen += not seen
        return action

    # ---- salvar / carregar ----
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"obs_type": self.obs_type, "alpha": self.alpha, "gamma": self.gamma, "q": self.q}
        with open(path, "wb") as f:  # "wb" = escrever bytes (pickle é um formato binário do Python)
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "QLearningAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(data["obs_type"], data["alpha"], data["gamma"])
        agent.q = data["q"]
        return agent
