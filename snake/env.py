"""Motor do jogo Snake com obstáculos dinâmicos (Fase 1).

Este arquivo contém SÓ as regras do jogo: nada de tela, nada de machine
learning. Isso segue a seção 9 da especificação (ambiente separado do
agente) e deixa o motor rápido para simular milhares de partidas.

Como usar (exemplo mínimo):

    from snake.env import SnakeGame, RIGHT
    game = SnakeGame(seed=42)
    result = game.step(RIGHT)      # anda uma casa para a direita
    print(game)                     # desenha o tabuleiro em texto
    print(result.game_over, result.end_reason)

Convenções:
- Coordenadas são tuplas (linha, coluna), começando em 0. A linha 0 é a de
  cima. Ex.: (4, 3) = 5ª linha, 4ª coluna.
- A cobra é guardada da cabeça para a cauda: snake[0] é a cabeça.
"""

# Permite usar anotações de tipo modernas (ex.: `int | None`) em qualquer lugar.
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# Apelido de tipo: "Cell" é só um nome mais legível para tuple[int, int].
# (Em TypeScript seria: type Cell = [number, number];)
Cell = tuple[int, int]

# ---------------------------------------------------------------------------
# Direções
# ---------------------------------------------------------------------------
# Usamos números (0 a 3) porque, mais tarde, o agente de RL vai escolher
# ações como números. Em JS seria um "enum" feito com constantes.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRECTION_NAMES = ("up", "right", "down", "left")

# Quanto a linha e a coluna mudam ao andar em cada direção.
# Lembre: a linha cresce para BAIXO, por isso UP é -1 na linha.
DIRECTION_DELTAS: dict[int, Cell] = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# ---------------------------------------------------------------------------
# Motivos de fim de partida
# ---------------------------------------------------------------------------
WALL = "wall"            # bateu na borda do mapa
BODY = "body"            # bateu no próprio corpo
OBSTACLE = "obstacle"    # bateu num obstáculo
TIMEOUT = "timeout"      # passou do limite de passos sem comer
WIN = "win"              # "zerou": não há célula livre para repor fruta

# As 8 vizinhas de uma célula (inclui diagonais). Usado na regra 4.2.
NEIGHBORS_8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


@dataclass(frozen=True)
class GameConfig:
    """Parâmetros do jogo. Os valores padrão seguem a especificação + CLAUDE.md.

    `@dataclass` gera o construtor automaticamente a partir dos campos
    (parecido com um objeto de configuração em JS: { height: 9, width: 10 }).
    `frozen=True` torna o objeto imutável, para ninguém alterar a config
    no meio de uma partida.

    Atenção: a frequência de obstáculos (1 a cada quantidade ímpar de frutas)
    NÃO é configurável de propósito — a especificação (seção 11) proíbe mudá-la.
    """

    height: int = 9
    width: int = 10
    num_fruits: int = 5
    # Cobra inicial, da cabeça para a cauda: cabeça em (4,3), corpo em (4,2) e (4,1).
    initial_snake: tuple[Cell, ...] = ((4, 3), (4, 2), (4, 1))
    initial_direction: int = RIGHT
    # Frutas iniciais fixas (decisão do CLAUDE.md). Só as reposições são sorteadas.
    initial_fruits: tuple[Cell, ...] = ((2, 4), (2, 8), (4, 6), (6, 4), (6, 8))
    # Células que a cobra cresce por fruta (a especificação, seção 6.2, pede
    # que seja um parâmetro; a decisão confirmada é +1).
    growth_per_fruit: int = 1
    # Profundidade da região de segurança à frente da cabeça (seção 4.3).
    safety_zone_length: int = 3
    # Limite de passos sem comer. None = sem limite.
    max_steps_without_fruit: int | None = 200


@dataclass
class StepResult:
    """O que aconteceu em um passo (útil para agentes, replays e testes)."""

    ate_fruit: bool = False
    new_fruit: Cell | None = None          # onde nasceu a fruta de reposição
    new_obstacle: Cell | None = None       # onde nasceu o obstáculo (se nasceu)
    game_over: bool = False
    end_reason: str | None = None


class SnakeGame:
    """Estado e regras de uma partida de Snake com obstáculos dinâmicos.

    Equivale a uma `class` do JavaScript. `self` é o `this` do JS, mas em
    Python ele aparece explicitamente como primeiro parâmetro dos métodos.
    """

    def __init__(self, config: GameConfig | None = None, seed: int | None = None):
        self.config = config or GameConfig()
        self._validate_config()
        self.reset(seed)

    # ------------------------------------------------------------------
    # Início de partida
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Começa uma partida nova.

        Se `seed` for informada, o gerador de números aleatórios é recriado
        com ela: mesma seed + mesmas ações = exatamente a mesma partida.
        Se não for, o gerador continua de onde parou (útil para jogar várias
        partidas seguidas sem repetir).
        """
        if seed is not None or not hasattr(self, "rng"):
            self.seed = seed
            # Gerador de aleatoriedade próprio da partida (não o global),
            # para que nada de fora interfira na reprodutibilidade.
            self.rng = np.random.default_rng(seed)

        cfg = self.config
        self._set_state(
            snake=cfg.initial_snake,
            direction=cfg.initial_direction,
            fruits=cfg.initial_fruits,
            obstacles=(),
            fruits_collected=0,
        )

    def load_state(
        self,
        snake: list[Cell] | tuple[Cell, ...],
        direction: int,
        fruits: list[Cell] | tuple[Cell, ...],
        obstacles: list[Cell] | tuple[Cell, ...] = (),
        fruits_collected: int = 0,
    ) -> None:
        """Monta um estado arbitrário (para testes, depuração e replays).

        O gerador aleatório NÃO é reiniciado aqui.
        """
        self._set_state(snake, direction, fruits, obstacles, fruits_collected)

    def _set_state(self, snake, direction, fruits, obstacles, fruits_collected) -> None:
        # O nome começa com "_" por convenção: é um método "interno" (em JS
        # seria algo como um #metodoPrivado).
        self.snake: deque[Cell] = deque(tuple(c) for c in snake)
        self.direction: int = direction
        self.fruits: list[Cell] = [tuple(c) for c in fruits]
        self.obstacles: list[Cell] = [tuple(c) for c in obstacles]
        self.fruits_collected: int = fruits_collected
        self.obstacles_generated: int = len(self.obstacles)
        # Quantos obstáculos já vieram prontos (só != 0 em cenários do load_state).
        self.obstacles_at_start: int = len(self.obstacles)
        self.steps: int = 0
        self.steps_since_fruit: int = 0
        self.pending_growth: int = 0
        self.game_over: bool = False
        self.end_reason: str | None = None
        self._validate_state()

    # ------------------------------------------------------------------
    # Um passo do jogo
    # ------------------------------------------------------------------
    def step(self, action: int | None = None) -> StepResult:
        """Avança um passo. `action` é a direção desejada (UP/RIGHT/DOWN/LEFT).

        `None` mantém a direção atual. Um comando de 180° (ex.: ir para a
        esquerda andando para a direita) é ignorado.

        Ordem dentro do passo (decisão do CLAUDE.md):
        mover -> checar colisão -> comer -> repor fruta -> gerar obstáculo.
        """
        if self.game_over:
            raise RuntimeError("A partida já acabou. Chame reset() para começar outra.")

        result = StepResult()

        # 1) Decide a direção (ignora meia-volta).
        if action is not None:
            if action not in DIRECTION_DELTAS:
                raise ValueError(f"Ação inválida: {action}. Use 0 (up), 1 (right), 2 (down) ou 3 (left).")
            if action != OPPOSITE[self.direction]:
                self.direction = action

        # 2) Calcula para onde a cabeça vai.
        dr, dc = DIRECTION_DELTAS[self.direction]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)
        self.steps += 1

        # 3) Checa colisões.
        eating = new_head in self.fruits
        # A cauda só fica parada se a cobra estiver crescendo neste passo.
        # Se ela não cresce, a cauda sai do lugar ao mesmo tempo em que a cabeça
        # anda — por isso entrar na casa da cauda é permitido (decisão confirmada).
        tail_stays = self.pending_growth > 0 or (eating and self.config.growth_per_fruit > 0)
        body_to_check = self.snake if tail_stays else list(self.snake)[:-1]

        if not self._in_bounds(new_head):
            return self._end(result, WALL)
        if new_head in self.obstacles:
            return self._end(result, OBSTACLE)
        if new_head in body_to_check:
            return self._end(result, BODY)

        # 4) Move: a cabeça entra na frente; a cauda sai, a não ser que esteja crescendo.
        #    (deque é uma lista otimizada para inserir/remover nas pontas,
        #    como um array do JS em que unshift() e pop() são baratos.)
        self.snake.appendleft(new_head)
        if eating:
            self.pending_growth += self.config.growth_per_fruit
        if self.pending_growth > 0:
            self.pending_growth -= 1
        else:
            self.snake.pop()

        # 5) Come a fruta, repõe e (se o total for ímpar) tenta gerar obstáculo.
        if eating:
            result.ate_fruit = True
            self.fruits.remove(new_head)
            self.fruits_collected += 1
            self.steps_since_fruit = 0

            new_fruit = self._spawn_fruit()
            if new_fruit is None:
                # Sem célula livre para repor a fruta = vitória (CLAUDE.md).
                return self._end(result, WIN)
            result.new_fruit = new_fruit

            if self.fruits_collected % 2 == 1:
                result.new_obstacle = self.try_spawn_obstacle()
        else:
            self.steps_since_fruit += 1
            limit = self.config.max_steps_without_fruit
            if limit is not None and self.steps_since_fruit >= limit:
                return self._end(result, TIMEOUT)

        return result

    def _end(self, result: StepResult, reason: str) -> StepResult:
        self.game_over = True
        self.end_reason = reason
        result.game_over = True
        result.end_reason = reason
        return result

    # ------------------------------------------------------------------
    # Frutas
    # ------------------------------------------------------------------
    def empty_cells(self) -> list[Cell]:
        """Células sem cobra, fruta ou obstáculo, em ordem fixa (linha a linha).

        A ordem fixa importa: o sorteio escolhe um ÍNDICE desta lista, então
        a lista precisa sair sempre igual para a seed reproduzir a partida.
        """
        occupied = set(self.snake) | set(self.fruits) | set(self.obstacles)
        return [
            (r, c)
            for r in range(self.config.height)
            for c in range(self.config.width)
            if (r, c) not in occupied
        ]

    def _spawn_fruit(self) -> Cell | None:
        """Põe uma fruta numa célula vazia sorteada de forma uniforme."""
        candidates = self.empty_cells()
        if not candidates:
            return None
        cell = candidates[self.rng.integers(len(candidates))]
        self.fruits.append(cell)
        return cell

    # ------------------------------------------------------------------
    # Obstáculos (seções 3, 4 e 5 da especificação)
    # ------------------------------------------------------------------
    def safety_zone(self) -> list[Cell]:
        """As até 3 células logo à frente da cabeça, na direção atual (seção 4.3).

        Células fora do mapa são simplesmente ignoradas.
        """
        dr, dc = DIRECTION_DELTAS[self.direction]
        head_r, head_c = self.snake[0]
        zone = []
        for k in range(1, self.config.safety_zone_length + 1):
            cell = (head_r + k * dr, head_c + k * dc)
            if self._in_bounds(cell):
                zone.append(cell)
        return zone

    def eligible_obstacle_cells(self) -> list[Cell]:
        """Todas as células onde um obstáculo novo poderia nascer agora.

        Uma célula é válida só se atender TODAS as regras ao mesmo tempo:
        - 4.1 está vazia (sem cobra, fruta ou obstáculo);
        - 4.2 nenhuma das 8 vizinhas é obstáculo;
        - 4.3 não está nas 3 células à frente da cabeça.
        (4.4: bordas são permitidas, então não há regra extra para elas.)
        """
        obstacles = set(self.obstacles)
        protected = set(self.safety_zone())
        eligible = []
        for r, c in self.empty_cells():
            if (r, c) in protected:
                continue
            touches_obstacle = any((r + dr, c + dc) in obstacles for dr, dc in NEIGHBORS_8)
            if touches_obstacle:
                continue
            eligible.append((r, c))
        return eligible

    def try_spawn_obstacle(self) -> Cell | None:
        """Tenta criar UM obstáculo numa posição válida sorteada (seção 5).

        Se não houver posição válida, nada acontece e o jogo segue (seção 5.1).
        Devolve a célula criada, ou None.
        """
        candidates = self.eligible_obstacle_cells()
        if not candidates:
            return None
        cell = candidates[self.rng.integers(len(candidates))]
        self.obstacles.append(cell)
        self.obstacles_generated += 1
        return cell

    # ------------------------------------------------------------------
    # Consultas para os agentes (seção 8)
    # ------------------------------------------------------------------
    @property
    def head(self) -> Cell:
        # @property deixa acessar como atributo: game.head (sem parênteses),
        # igual a um "get head()" de uma classe JS.
        return self.snake[0]

    def safe_directions(self) -> list[int]:
        """Direções que NÃO causam colisão imediata no próximo passo.

        Útil para agentes (e, mais tarde, para "máscaras de ação" no RL).
        Uma meia-volta nunca aparece aqui, pois é ignorada pelo jogo.
        """
        safe = []
        for direction, (dr, dc) in DIRECTION_DELTAS.items():
            if direction == OPPOSITE[self.direction]:
                continue
            cell = (self.head[0] + dr, self.head[1] + dc)
            eating = cell in self.fruits
            tail_stays = self.pending_growth > 0 or (eating and self.config.growth_per_fruit > 0)
            body = self.snake if tail_stays else list(self.snake)[:-1]
            if self._in_bounds(cell) and cell not in self.obstacles and cell not in body:
                safe.append(direction)
        return safe

    def get_state(self) -> dict:
        """Retrato do estado atual com as variáveis da seção 8 da especificação.

        Devolve cópias, para que quem consulta não altere o jogo sem querer.
        """
        return {
            "snake_body": list(self.snake),
            "snake_head": self.head,
            "snake_direction": self.direction,
            "fruits": list(self.fruits),
            "obstacles": list(self.obstacles),
            "fruits_collected": self.fruits_collected,
            "obstacles_generated": self.obstacles_generated,
            "board_width": self.config.width,
            "board_height": self.config.height,
            "game_over": self.game_over,
            # Extras úteis para métricas e depuração:
            "end_reason": self.end_reason,
            "steps": self.steps,
            "steps_since_fruit": self.steps_since_fruit,
        }

    def __str__(self) -> str:
        """Desenho do tabuleiro em texto (o print() usa isto, como o toString() do JS).

        H = cabeça, o = corpo, * = fruta, # = obstáculo, . = vazio.
        """
        grid = [["." for _ in range(self.config.width)] for _ in range(self.config.height)]
        for r, c in self.obstacles:
            grid[r][c] = "#"
        for r, c in self.fruits:
            grid[r][c] = "*"
        for r, c in list(self.snake)[1:]:
            grid[r][c] = "o"
        if self._in_bounds(self.head):
            grid[self.head[0]][self.head[1]] = "H"
        return "\n".join("".join(row) for row in grid)

    # ------------------------------------------------------------------
    # Auxiliares e validações
    # ------------------------------------------------------------------
    def _in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.config.height and 0 <= c < self.config.width

    def _validate_config(self) -> None:
        cfg = self.config
        if len(cfg.initial_fruits) != cfg.num_fruits:
            raise ValueError("initial_fruits precisa ter exatamente num_fruits posições.")
        if len(cfg.initial_snake) < 1:
            raise ValueError("A cobra inicial precisa ter pelo menos a cabeça.")
        if cfg.initial_direction not in DIRECTION_DELTAS:
            raise ValueError("initial_direction inválida.")

    def _validate_state(self) -> None:
        """Garante que o estado montado é coerente (nada fora do mapa ou sobreposto)."""
        cells = list(self.snake) + self.fruits + self.obstacles
        for cell in cells:
            if not self._in_bounds(cell):
                raise ValueError(f"Célula fora do tabuleiro: {cell}")
        if len(set(cells)) != len(cells):
            raise ValueError("Cobra, frutas e obstáculos não podem dividir a mesma célula.")
        if self.direction not in DIRECTION_DELTAS:
            raise ValueError(f"Direção inválida: {self.direction}")
