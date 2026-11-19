"""DQN (Deep Q-Network), Fase 5: o Q-learning com uma rede neural no lugar da tabela.

Na Fase 4, a tabela guardava uma nota Q por situação e só sabia responder sobre
situações que já tinha visto. Aqui uma REDE NEURAL recebe a observação e devolve
as 4 notas de uma vez:

    observação (18 números ou o tabuleiro) -> rede -> [Q(cima), Q(direita), Q(baixo), Q(esquerda)]

Como a rede é uma função "contínua", situações parecidas recebem notas parecidas,
mesmo que ela nunca tenha visto aquela exata. Isso é GENERALIZAR.

O aprendizado continua sendo o do Q-learning, só que em vez de "nota += alpha * erro"
a gente ajusta os pesos da rede para diminuir o erro (gradiente descendente):

    alvo = recompensa + gamma * max Q_alvo(s')      (0 no futuro se a partida acabou)
    erro = Q(s, a) - alvo  ->  a rede é empurrada para reduzir esse erro

Os dois truques que fazem o DQN funcionar (DeepMind, 2015):
1. Replay buffer: guarda as últimas N jogadas e treina com lotes SORTEADOS delas.
   Jogadas seguidas são muito parecidas entre si; sortear quebra essa correlação
   e reaproveita cada experiência várias vezes.
2. Rede-alvo: uma cópia "congelada" da rede, atualizada só de tempos em tempos,
   usada para calcular o alvo. Sem ela, a rede persegue um alvo que muda a cada
   passo (como um cachorro correndo atrás do próprio rabo).

PyTorch em 3 linhas, comparando com JS: um `nn.Module` é como uma classe com um
método `forward(x)`; um `torch.Tensor` é um array numpy que pode morar na placa
de vídeo (GPU) e que "lembra" as contas feitas, para calcular o gradiente sozinho.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from snake.env import SnakeGame
from snake.gym_env import CHANNELS, OBS_SIZES, OBSERVERS


# ---------------------------------------------------------------------------
# As redes
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Rede "densa" para vetores de características.

    Exemplo com 18 entradas: 18 -> 128 -> 128 -> 4. Cada "->" é uma camada Linear
    (cada saída é uma soma ponderada de todas as entradas) seguida de ReLU
    (zera os negativos; é o que permite à rede aprender coisas não lineares).
    """

    def __init__(self, n_inputs: int, hidden: int = 128, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),  # a última não tem ReLU: notas podem ser negativas
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    """Rede convolucional para o tabuleiro em canais (4 × linhas × colunas).

    Uma convolução 3×3 é um "carimbo" que desliza pelo tabuleiro olhando cada
    casa com suas 8 vizinhas, e aprende padrões locais (parede ao lado, corpo
    em L...). Duas camadas seguidas enxergam 5×5 em volta de cada casa. Sem
    "pooling": o tabuleiro já é pequeno e a posição exata importa.
    """

    def __init__(self, height: int, width: int, n_channels: int = len(CHANNELS), n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1), nn.ReLU(),  # padding=1: mantém 9×10
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),  # (64, 9, 10) -> 5.760 números em fila
            nn.Linear(64 * height * width, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())  # o tabuleiro chega como bytes 0/1 (uint8)


def build_network(obs_type: str, height: int, width: int) -> nn.Module:
    if obs_type == "grid":
        return CNN(height, width)
    return MLP(OBS_SIZES[obs_type])


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Memória circular das últimas `capacity` jogadas (s, a, r, s', fim).

    "Circular" = quando enche, a jogada nova sobrescreve a mais velha, como uma
    fila de tamanho fixo. Os arrays são criados de uma vez (mais rápido do que
    uma lista de objetos, que no JS seria um array de {s, a, r, ...}).
    """

    def __init__(self, capacity: int, obs_shape: tuple, obs_dtype, seed: int | None = None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.pos = 0     # onde entra a próxima jogada
        self.size = 0    # quantas jogadas válidas existem
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action: int, reward: float, next_obs, terminated: bool) -> None:
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, terminated
        self.pos = (self.pos + 1) % self.capacity  # % faz a volta ao início
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {"obs": self.obs[idx], "actions": self.actions[idx], "rewards": self.rewards[idx],
                "next_obs": self.next_obs[idx], "terminated": self.terminated[idx]}

    def __len__(self) -> int:  # permite len(buffer), como .length no JS
        return self.size


# ---------------------------------------------------------------------------
# O agente
# ---------------------------------------------------------------------------
class DQNAgent:
    name = "dqn"

    def __init__(self, obs_type: str, height: int, width: int, gamma: float = 0.95, lr: float = 5e-4,
                 device: str = "cpu", seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)  # mesmos pesos iniciais com a mesma seed
        self.obs_type = obs_type
        self.observe = OBSERVERS[obs_type]
        self.height, self.width = height, width
        self.gamma = gamma
        self.lr = lr
        self.device = torch.device(device)
        self.q_net = build_network(obs_type, height, width).to(self.device)
        self.target_net = build_network(obs_type, height, width).to(self.device)
        self.sync_target()
        # Adam: o "ajustador" de pesos mais usado; lr = tamanho de cada passinho.
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

    # ---- decidir ----
    @torch.no_grad()  # "decorador": aqui não precisa guardar contas para gradiente (mais rápido)
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, device=self.device).unsqueeze(0)  # lote de 1: (1, ...)
        return self.q_net(x)[0].cpu().numpy()

    def choose(self, obs: np.ndarray, epsilon: float) -> int:
        """epsilon-greedy, igual à Fase 4: sorteia com prob. epsilon, senão a melhor nota."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(4))
        return int(np.argmax(self.q_values(obs)))

    # ---- aprender ----
    def sync_target(self) -> None:
        """Copia os pesos da rede principal para a rede-alvo."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train_step(self, batch: dict[str, np.ndarray]) -> float:
        """Um passo de gradiente num lote de jogadas. Devolve o erro (loss) médio."""
        t = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        # Q(s, a) só da ação que foi de fato jogada em cada linha do lote.
        q = self.q_net(t["obs"]).gather(1, t["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            future = self.target_net(t["next_obs"]).max(dim=1).values
            target = t["rewards"] + self.gamma * future * (1.0 - t["terminated"])
        # Huber: igual ao erro quadrático para erros pequenos, mas cresce só em linha
        # reta para erros grandes (um alvo muito errado não "explode" os pesos).
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()   # zera os gradientes do passo anterior
        loss.backward()              # calcula o gradiente de cada peso (o PyTorch faz sozinho)
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)  # trava passos gigantes
        self.optimizer.step()        # anda um passinho na direção que reduz o erro
        return float(loss.item())

    # ---- interface de agente (igual às Fases 3 e 4, para o evaluate) ----
    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, game: SnakeGame) -> int:
        return self.choose(self.observe(game), epsilon=0.0)  # avaliação: sem explorar

    def action_scores(self, game: SnakeGame) -> np.ndarray:
        """As 4 notas Q (a busca da Fase 7 usa para saber a 2ª, 3ª... opção)."""
        return self.q_values(self.observe(game))

    # ---- salvar / carregar ----
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"obs_type": self.obs_type, "height": self.height, "width": self.width,
                    "gamma": self.gamma, "lr": self.lr, "weights": self.q_net.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "DQNAgent":
        data = torch.load(path, map_location=device)
        agent = cls(data["obs_type"], data["height"], data["width"], data["gamma"], data["lr"], device=device)
        agent.q_net.load_state_dict(data["weights"])
        agent.sync_target()
        return agent
