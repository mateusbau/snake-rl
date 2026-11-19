"""PPO (Fase 6) via Stable-Baselines3: o algoritmo "de prateleira" mais usado em RL.

Diferença de ideia para o DQN (Fase 5):

- O DQN aprende NOTAS (Q) e joga escolhendo a maior. É "indireto".
- O PPO aprende direto a POLÍTICA: uma rede que devolve probabilidades das 4 ações,
  ex.: [cima 0,70, direita 0,25, baixo 0,00, esquerda 0,05]. Treinar = aumentar a
  probabilidade das ações que deram mais certo do que o esperado e diminuir as outras.
  "Mais certo do que o esperado" vem de uma segunda cabeça da rede, o CRÍTICO, que
  estima quanto a partida ainda vai render a partir daqui (o "valor" do estado).
- O "P" de Proximal: cada atualização só pode mudar as probabilidades um pouco
  (clip de 20%). Isso deixa o treino estável: sem saltos que estragam o que já aprendeu.
- Ele joga em VÁRIOS tabuleiros ao mesmo tempo (ambientes em paralelo), junta um
  "lote" de experiência fresca, aprende com ele e joga fora. Não há replay buffer.

MaskablePPO (sb3-contrib) acrescenta a MÁSCARA: antes de sortear a ação, as que
matam na hora recebem probabilidade 0. A rede não perde tempo aprendendo "não bata
na parede" e foca no que importa: não se encurralar e comer.

A biblioteca é como um framework em JS (React, Express): a gente entrega o ambiente
e os hiperparâmetros, ela cuida do laço de treino.
"""

from __future__ import annotations

import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from snake.boards import BOARDS
from snake.env import SnakeGame
from snake.gym_env import OBSERVERS, SnakeEnv, action_mask

# Hiperparâmetros padrão do projeto (cada mudança = um experimento, regra 4).
PPO_DEFAULTS = dict(
    learning_rate=3e-4,
    n_steps=512,        # passos por ambiente antes de cada atualização (8 ambientes -> lote de 4.096)
    batch_size=512,     # o lote é dividido em pedaços de 512 para o gradiente
    n_epochs=4,         # cada lote é reaproveitado 4 vezes e depois descartado
    gamma=0.99,         # horizonte mais longo que o DQN (0,95): o fim de jogo é longe
    gae_lambda=0.95,
    clip_range=0.2,     # o "Proximal": no máximo ~20% de mudança por atualização
    ent_coef=0.01,      # bônus por "dúvida": evita decidir cedo demais e parar de explorar
)
NET_ARCH = dict(pi=[128, 128], vf=[128, 128])  # mesmo tamanho do MLP do DQN, para comparar


class EnvFactory:
    """Cria um SnakeEnv. É uma classe (e não uma lambda) porque os ambientes em
    processos separados recebem a "receita" por pickle, e lambdas não são picláveis."""

    def __init__(self, board: str, obs_type: str, start_states: list[dict] | None = None, start_prob: float = 0.0):
        self.board, self.obs_type = board, obs_type
        self.start_states, self.start_prob = start_states, start_prob

    def __call__(self) -> SnakeEnv:
        return SnakeEnv(BOARDS[self.board], obs_type=self.obs_type,
                        start_states=self.start_states, start_prob=self.start_prob)


def build_model(env, masked: bool, seed: int | None = None, **overrides):
    algo = MaskablePPO if masked else PPO
    params = {**PPO_DEFAULTS, **overrides}
    return algo("MlpPolicy", env, policy_kwargs={"net_arch": NET_ARCH}, seed=seed, device="cpu", verbose=0, **params)


class PPOAgent:
    """Embrulha um modelo do SB3 na interface de agente do projeto (reset/act)."""

    name = "ppo"

    def __init__(self, model, obs_type: str, masked: bool, board: str = "full"):
        self.model = model
        self.obs_type = obs_type
        self.observe = OBSERVERS[obs_type]
        self.masked = masked
        self.board = board
        self.config = BOARDS[board]

    @classmethod
    def new(cls, obs_type: str, masked: bool, seed: int | None = None, board: str = "full") -> "PPOAgent":
        env = DummyVecEnv([EnvFactory(board, obs_type)])
        return cls(build_model(env, masked, seed), obs_type, masked, board)

    def reset(self, seed: int | None = None) -> None:
        pass  # deterministic=True: a política não sorteia nada na avaliação

    def act(self, game: SnakeGame) -> int:
        obs = self.observe(game)
        if self.masked:
            action, _ = self.model.predict(obs, deterministic=True, action_masks=action_mask(game))
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    # ---- salvar / carregar: o .zip do SB3 + um .json com o que o projeto precisa ----
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        info = {"obs_type": self.obs_type, "masked": self.masked, "board": self.board}
        path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PPOAgent":
        path = Path(path)
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        algo = MaskablePPO if info["masked"] else PPO
        return cls(algo.load(path, device="cpu"), info["obs_type"], info["masked"], info.get("board", "full"))


def train_quick(board: str, obs_type: str, steps: int, masked: bool, seed: int = 0) -> PPOAgent:
    """Treino pequeno, num só processo (usado nos testes)."""
    env = DummyVecEnv([EnvFactory(board, obs_type) for _ in range(4)])
    env.seed(seed)
    model = build_model(env, masked, seed, n_steps=256, batch_size=256)
    model.learn(total_timesteps=steps)
    return PPOAgent(model, obs_type, masked, board)
