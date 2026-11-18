"""O jogo no formato Gymnasium (Fase 4): a "tomada padrão" das bibliotecas de RL.

Gymnasium é o padrão que quase toda biblioteca de RL entende (Stable-Baselines3,
que usaremos na Fase 6, inclusive). Todo ambiente Gymnasium tem a mesma cara:

    obs, info = env.reset(seed=42)                 # começa uma partida
    obs, reward, terminated, truncated, info = env.step(action)

- obs (observação): o que o agente "vê" (aqui, um array numpy).
- reward (recompensa): o número que o agente tenta maximizar ao longo do tempo.
- terminated: a partida acabou "de verdade" (morreu ou zerou).
- truncated: a partida foi cortada por um limite externo (aqui, o limite de passos
  sem comer). A diferença importa para o aprendizado: veja fase-4.md.

Este arquivo só ADAPTA o SnakeGame; as regras continuam todas em env.py (seção 9).
É como um "adapter" no JavaScript: embrulha um objeto para ele falar outra interface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snake.env import DIRECTION_DELTAS, TIMEOUT, WIN, GameConfig, SnakeGame

# Canais da observação "grid" (um tabuleiro por canal, como as camadas de cor de uma imagem).
CHANNELS = ("head", "body", "fruits", "obstacles")
FEATURE_NAMES = (
    "danger_straight", "danger_right", "danger_left",
    "dir_up", "dir_right", "dir_down", "dir_left",
    "fruit_up", "fruit_down", "fruit_left", "fruit_right",
)


@dataclass(frozen=True)
class Rewards:
    """Quanto vale cada acontecimento. Mudar isso é um experimento (regra 4 do CLAUDE.md)."""

    fruit: float = 1.0
    death: float = -1.0
    win: float = 10.0
    step: float = 0.0       # custo por passo (0 = sem pressa)
    timeout: float = 0.0    # o corte por tempo não é "culpa" igual a morrer


# ---------------------------------------------------------------------------
# Observações: funções puras (recebem o jogo e devolvem um array)
# ---------------------------------------------------------------------------
def grid_observation(game: SnakeGame) -> np.ndarray:
    """Tabuleiro em 4 canais 0/1, formato (canais, linhas, colunas).

    Exemplo no 5×5 (cobra H o o na linha 2, fruta em (2,4)):
        canal head:   só (2,2) = 1
        canal body:   (2,1) e (2,0) = 1
        canal fruits: (2,4) = 1
        canal obstacles: tudo 0
    """
    cfg = game.config
    obs = np.zeros((len(CHANNELS), cfg.height, cfg.width), dtype=np.uint8)
    head, *body = game.snake  # "desestruturação", como const [head, ...body] = snake no JS
    obs[0][head] = 1
    for cell in body:
        obs[1][cell] = 1
    for cell in game.fruits:
        obs[2][cell] = 1
    for cell in game.obstacles:
        obs[3][cell] = 1
    return obs


def feature_observation(game: SnakeGame) -> np.ndarray:
    """Resumo em 11 números 0/1, independente do tamanho do tabuleiro.

    - perigo à frente / à direita / à esquerda (relativo à direção atual);
    - direção atual (4 posições, só uma vale 1: "one-hot");
    - onde está a fruta mais próxima: acima / abaixo / à esquerda / à direita.
    """
    direction = game.direction
    safe = set(game.safe_directions())
    # UP=0, RIGHT=1, DOWN=2, LEFT=3 em sentido horário: +1 vira à direita, +3 à esquerda.
    relative = (direction, (direction + 1) % 4, (direction + 3) % 4)
    danger = [d not in safe for d in relative]
    one_hot = [direction == d for d in range(4)]

    hr, hc = game.head
    if game.fruits:
        fr, fc = min(game.fruits, key=lambda f: abs(f[0] - hr) + abs(f[1] - hc))
        fruit = [fr < hr, fr > hr, fc < hc, fc > hc]
    else:
        fruit = [False] * 4
    return np.array(danger + one_hot + fruit, dtype=np.float32)


SPACE_FEATURE_NAMES = FEATURE_NAMES + (
    "space_straight", "space_right", "space_left",
    "tail_straight", "tail_right", "tail_left",
    "length",
)


def space_feature_observation(game: SnakeGame) -> np.ndarray:
    """Os 11 números de antes + 7 que "enxergam" o corpo (Fase 5): 18 no total.

    O resumo de 11 números só olha 1 casa em volta da cabeça, e por isso a cobra
    entra em "bolsões" formados pelo próprio corpo e morre lá dentro (59% das
    mortes na Fase 4). Os números novos olham mais longe, simulando cada jogada:

    - space_X: depois de ir para X, quanto do espaço livre a cabeça ainda alcança
      (0 a 1; é o flood fill da Fase 3). 0 também quando X mata na hora.
    - tail_X: depois de ir para X, a cabeça ainda alcança a cauda? (0/1; o teste
      de segurança do agente BFS.)
    - length: tamanho da cobra / células do tabuleiro (0 a 1).

    Exemplo: cobra num corredor que vira beco à direita -> space_right ≈ 0,05
    (5% do espaço) e space_left ≈ 0,90. A rede aprende sozinha que 0,05 é ruim.

    Importante: isso NÃO diz à rede o que fazer (o agente BFS decide com regras);
    só entrega mais informação. Quem decide o que fazer com ela é o treino.
    """
    # Importado aqui dentro para o motor do jogo não depender da pasta agents/
    # quando ninguém usa esta observação.
    from agents.grid import flood_fill, tail_reachable

    cfg = game.config
    direction = game.direction
    safe = set(game.safe_directions())
    body = list(game.snake)
    obstacles = set(game.obstacles)
    fruits = set(game.fruits)
    hr, hc = game.head
    space, tail = [], []
    for d in (direction, (direction + 1) % 4, (direction + 3) % 4):  # frente, direita, esquerda
        if d not in safe:
            space.append(0.0)
            tail.append(0.0)
            continue
        dr, dc = DIRECTION_DELTAS[d]
        new_head = (hr + dr, hc + dc)
        # Se come, a cauda fica; senão, a cauda sai (mesma regra do motor).
        new_body = [new_head] + (body if new_head in fruits else body[:-1])
        free = cfg.height * cfg.width - len(new_body) - len(obstacles)
        reach = flood_fill(new_head, obstacles | set(new_body[1:]), cfg.height, cfg.width)
        space.append(min(1.0, reach / free) if free > 0 else 0.0)
        tail.append(float(tail_reachable(new_body, obstacles, cfg.height, cfg.width)))
    length = len(body) / (cfg.height * cfg.width)
    return np.concatenate([feature_observation(game), np.array(space + tail + [length], dtype=np.float32)])


OBSERVERS = {"grid": grid_observation, "features": feature_observation, "features_space": space_feature_observation}
OBS_SIZES = {"features": len(FEATURE_NAMES), "features_space": len(SPACE_FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# O ambiente
# ---------------------------------------------------------------------------
class SnakeEnv(gym.Env):
    """Snake com obstáculos no formato Gymnasium.

    Ações: Discrete(4) = 0 cima, 1 direita, 2 baixo, 3 esquerda (as mesmas do motor;
    meia-volta é ignorada, ou seja, vale como "seguir reto").
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: GameConfig | None = None, obs_type: str = "grid",
                 rewards: Rewards | None = None, render_mode: str | None = None):
        super().__init__()
        if obs_type not in OBSERVERS:
            raise ValueError(f"obs_type deve ser um de {list(OBSERVERS)}")
        self.config = config or GameConfig()
        self.obs_type = obs_type
        self.observe = OBSERVERS[obs_type]
        self.rewards = rewards or Rewards()
        self.render_mode = render_mode
        self.game = SnakeGame(self.config, seed=0)
        self._renderer = None

        self.action_space = spaces.Discrete(4)
        if obs_type == "grid":
            shape = (len(CHANNELS), self.config.height, self.config.width)
            self.observation_space = spaces.Box(0, 1, shape=shape, dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(0, 1, shape=(OBS_SIZES[obs_type],), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # super().reset(seed=...) prepara self.np_random, o gerador do próprio ambiente.
        super().reset(seed=seed)
        # Com seed: a partida é exatamente a do motor com essa seed (igual ao evaluate.py).
        # Sem seed: sorteia uma, de forma reproduzível a partir do gerador do ambiente.
        game_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.game = SnakeGame(self.config, seed=game_seed)
        return self.observe(self.game), self._info()

    def step(self, action):
        result = self.game.step(int(action))
        r = self.rewards
        reward = r.step
        if result.ate_fruit:
            reward += r.fruit
        terminated = truncated = False
        if self.game.game_over:
            if self.game.end_reason == WIN:
                reward += r.win
                terminated = True
            elif self.game.end_reason == TIMEOUT:
                reward += r.timeout
                truncated = True
            else:  # parede, corpo ou obstáculo
                reward += r.death
                terminated = True
        return self.observe(self.game), float(reward), terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        """Quais ações não matam no próximo passo (MaskablePPO usa isso na Fase 6)."""
        safe = set(self.game.safe_directions())
        return np.array([d in safe for d in DIRECTION_DELTAS], dtype=bool)

    def _info(self) -> dict:
        return {
            "fruits": self.game.fruits_collected,
            "steps": self.game.steps,
            "end_reason": self.game.end_reason,
            "seed": self.game.seed,
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # desenha sem abrir janela
        import pygame

        from snake.render import Renderer

        if self._renderer is None:
            pygame.init()
            self._renderer = Renderer(self.game)
            self._surface = pygame.Surface((self._renderer.width, self._renderer.height))
        self._renderer.draw(self._surface, self.game)
        # pygame guarda (largura, altura, cor); imagens no numpy são (altura, largura, cor).
        return np.transpose(pygame.surfarray.array3d(self._surface), (1, 0, 2))
