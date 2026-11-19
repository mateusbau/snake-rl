"""Fase 7: rede + busca. A rede sugere; a busca confere antes de jogar.

Duas camadas, que podem ser empilhadas por cima de qualquer agente:

1. SafetyAgent ("checagem de segurança"): pega as jogadas em ordem de preferência
   da rede (ou de quem estiver por baixo) e veta as que deixam a cabeça sem caminho
   até a cauda. É o teste do BFS (Fase 3), agora como "freio" de uma rede.

2. EndgameAgent ("solucionador de fim de jogo"): quando não sobra NENHUMA célula
   vazia, o jogo vira um quebra-cabeça 100% previsível. Não nasce fruta (não há
   onde) nem obstáculo (idem), e cada passo só pode ir para uma fruta ou para a
   casa da própria cauda. Então uma busca em profundidade (DFS) testa as
   sequências possíveis até achar uma que coma tudo, ou provar que não existe.

A busca usa um simulador PRÓPRIO, sem sorteio, e só é ligada quando não há sorteio
pela frente. Copiar o SnakeGame e simular seria "colar": a cópia levaria junto o
gerador de números e "veria" onde as próximas frutas vão nascer.
"""

from __future__ import annotations

import numpy as np

from agents.grid import flood_fill, tail_reachable
from snake.env import DIRECTION_DELTAS, OPPOSITE, SnakeGame

Cell = tuple[int, int]


def color_imbalance(obstacles, height: int, width: int) -> int:
    """|células livres "pretas" - "brancas"|, pintando o tabuleiro como xadrez.

    O corpo da cobra alterna as duas cores. Para zerar, ele cobre todas as livres,
    então a diferença tem que ser <= 1. Com 2 ou mais, zerar é impossível.
    """
    free = [0, 0]
    for r in range(height):
        for c in range(width):
            free[(r + c) % 2] += 1
    for r, c in obstacles:
        free[(r + c) % 2] -= 1
    return abs(free[0] - free[1])


def preferences(base, game: SnakeGame) -> list[int]:
    """As 4 direções da preferida para a menos preferida, segundo o agente `base`.

    Redes dão notas para as 4 (action_scores); agentes de regras (BFS...) só dizem
    a escolhida, que então vai para o topo e as outras ficam na ordem padrão.
    """
    if hasattr(base, "action_scores"):
        scores = np.asarray(base.action_scores(game), dtype=float)
        return [int(d) for d in np.argsort(-scores, kind="stable")]
    first = base.act(game)
    return [first] + [d for d in DIRECTION_DELTAS if d != first]


def body_after(game: SnakeGame, direction: int) -> list[Cell]:
    """Corpo depois de andar em `direction` (supondo que é seguro): come -> cresce."""
    hr, hc = game.head
    dr, dc = DIRECTION_DELTAS[direction]
    new_head = (hr + dr, hc + dc)
    body = list(game.snake)
    return [new_head] + (body if new_head in game.fruits else body[:-1])


class SafetyAgent:
    """Segue a preferência do agente de baixo, pulando jogadas que encurralam."""

    def __init__(self, base, name: str | None = None):
        self.base = base
        self.name = name or f"{base.name}+seguranca"
        self.vetoes = 0  # quantas vezes a busca discordou da rede (para o relatório)

    def reset(self, seed: int | None = None) -> None:
        self.base.reset(seed)

    def stats(self) -> dict:
        """Contadores para o relatório (somados com os de quem está por baixo)."""
        return {**getattr(self.base, "stats", dict)(), "vetoes": self.vetoes}

    def act(self, game: SnakeGame) -> int:
        cfg = game.config
        safe = set(game.safe_directions())
        prefs = preferences(self.base, game)
        if not safe:
            return prefs[0]  # presa: tanto faz
        obstacles = set(game.obstacles)
        for i, d in enumerate(prefs):
            if d in safe and tail_reachable(body_after(game, d), obstacles, cfg.height, cfg.width):
                self.vetoes += i > 0 and prefs[0] in safe
                return d
        # Nenhuma mantém a cauda alcançável: escolhe a que deixa mais espaço livre.
        self.vetoes += 1

        def space(d: int) -> int:
            body = body_after(game, d)
            return flood_fill(body[0], obstacles | set(body[1:]), cfg.height, cfg.width)

        return max(safe, key=space)


class EndgameAgent:
    """Quando não há célula vazia, procura uma sequência que zera; senão, usa o de baixo."""

    def __init__(self, base, name: str | None = None, budget: int = 200_000):
        self.base = base
        self.name = name or f"{base.name}+fim"
        self.budget = budget          # máximo de estados visitados por busca
        self.plan: list[int] = []
        self.solved = self.searches = self.proved_impossible = 0

    def reset(self, seed: int | None = None) -> None:
        self.base.reset(seed)
        self.plan = []

    def stats(self) -> dict:
        return {**getattr(self.base, "stats", dict)(), "endgame_searches": self.searches,
                "endgame_solved": self.solved, "endgame_impossible": self.proved_impossible}

    def act(self, game: SnakeGame) -> int:
        if self.plan:
            return self.plan.pop(0)
        if game.fruits and not game.empty_cells():
            self.searches += 1
            plan = solve_endgame(game, self.budget)
            if plan is None:
                self.proved_impossible += 1
            elif plan:
                self.solved += 1
                self.plan = plan[1:]
                return plan[0]
        return self.base.act(game)


class HunterAgent:
    """Lookahead "caçador": quando a cobra fica muito tempo sem comer, procura uma
    refeição segura por caminhos que NÃO precisam ser os mais curtos.

    Por que ajuda: o BFS só testa o caminho mais curto até cada fruta. Num caminho
    mais longo, a cauda anda mais até a cabeça chegar, e o corpo fica em outra
    posição na hora de comer. Às vezes é essa posição que deixa a cauda alcançável.

    A busca simula só até comer (sem sorteio pela frente): depois de comer nasce
    uma fruta nova em lugar desconhecido, e aí a gente decide de novo.
    """

    def __init__(self, base, name: str | None = None, trigger: int = 50, budget: int = 30_000):
        self.base = base
        self.name = name or f"{base.name}+caca"
        self.trigger = trigger      # passos sem comer antes de começar a caçar
        self.budget = budget        # máximo de estados por busca
        self.plan: list[int] = []
        self.searches = self.found = 0

    def reset(self, seed: int | None = None) -> None:
        self.base.reset(seed)
        self.plan = []

    def stats(self) -> dict:
        return {**getattr(self.base, "stats", dict)(), "hunt_searches": self.searches, "hunt_found": self.found}

    def act(self, game: SnakeGame) -> int:
        if self.plan:
            return self.plan.pop(0)
        if game.steps_since_fruit >= self.trigger and game.fruits:
            self.searches += 1
            plan = find_safe_meal(game, self.budget)
            if plan:
                self.found += 1
                self.plan = plan[1:]
                return plan[0]
        return self.base.act(game)


def find_safe_meal(game: SnakeGame, budget: int) -> list[int] | None:
    """DFS por um caminho que come uma fruta e, depois de comer, a cauda continua
    alcançável (ou o jogo zera). Tenta primeiro as direções que aproximam da fruta
    mais próxima. Devolve as jogadas, ou None se não achou dentro do orçamento."""
    cfg = game.config
    height, width = cfg.height, cfg.width
    cells = height * width
    limit = cfg.max_steps_without_fruit or 10**9
    max_depth = limit - game.steps_since_fruit - 1  # chegar antes de o tempo esgotar
    obstacles = frozenset(game.obstacles)
    obstacle_set = set(obstacles)
    fruits = frozenset(game.fruits)
    stack = [(tuple(game.snake), game.direction, [])]
    seen = set()
    while stack and len(seen) < budget:
        body, direction, path = stack.pop()
        if body in seen or len(path) >= max_depth:
            continue
        seen.add(body)
        hr, hc = body[0]
        children = []
        for d, (dr, dc) in DIRECTION_DELTAS.items():
            if d == OPPOSITE[direction]:
                continue
            head = (hr + dr, hc + dc)
            if not (0 <= head[0] < height and 0 <= head[1] < width) or head in obstacles:
                continue
            eats = head in fruits
            if head in (body if eats else body[:-1]):
                continue
            if eats:
                new_body = (head,) + body
                won = len(fruits) == 1 and len(new_body) + len(obstacles) == cells
                if won or tail_reachable(list(new_body), obstacle_set, height, width):
                    return path + [d]
                continue  # comer aqui encurrala: não segue por este ramo
            dist = min(abs(head[0] - f[0]) + abs(head[1] - f[1]) for f in fruits)
            children.append((dist, ((head,) + body[:-1], d, path + [d])))
        # Pilha: o último empilhado sai primeiro -> empilha os mais longe antes.
        for _, child in sorted(children, key=lambda c: -c[0]):
            stack.append(child)
    return None


def solve_endgame(game: SnakeGame, budget: int) -> list[int] | None | bool:
    """DFS pelas jogadas possíveis até comer todas as frutas.

    Devolve a lista de jogadas que zera; None se provou que é impossível (explorou
    tudo); False se estourou o orçamento sem decidir.
    """
    cfg = game.config
    height, width, limit = cfg.height, cfg.width, cfg.max_steps_without_fruit
    obstacles = frozenset(game.obstacles)
    start = (tuple(game.snake), game.direction, frozenset(game.fruits), game.steps_since_fruit)
    stack = [(start, [])]
    seen = set()
    visited = 0
    while stack:
        (body, direction, fruits, since), path = stack.pop()
        key = (body, fruits)
        if key in seen:
            continue  # já estivemos aqui: andar em círculo não leva a lugar novo
        seen.add(key)
        visited += 1
        if visited > budget:
            return False
        hr, hc = body[0]
        children = []
        for d, (dr, dc) in DIRECTION_DELTAS.items():
            if d == OPPOSITE[direction]:
                continue
            head = (hr + dr, hc + dc)
            if not (0 <= head[0] < height and 0 <= head[1] < width) or head in obstacles:
                continue
            eats = head in fruits
            if head in (body if eats else body[:-1]):
                continue
            if eats:
                new_fruits = fruits - {head}
                if not new_fruits:
                    return path + [d]  # comeu a última: cobra + obstáculos = tabuleiro todo
                children.append((0, ((head,) + body, d, new_fruits, 0), path + [d]))
            elif since + 1 < (limit or 10**9):
                children.append((1, ((head,) + body[:-1], d, fruits, since + 1), path + [d]))
        # A pilha tira o último primeiro: empilha "comer" por último para testar antes.
        for _, child, child_path in sorted(children, key=lambda c: -c[0]):
            stack.append((child, child_path))
    return None
