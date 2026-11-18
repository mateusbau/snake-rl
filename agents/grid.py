"""Ferramentas de "olhar o mapa" usadas pelos agentes sem ML (Fase 3).

Os agentes NÃO mexem nas regras do jogo (seção 9 da especificação): eles só leem
o estado (`game.get_state()`) e fazem contas próprias em cima dele, como um
jogador humano que olha a tela e planeja. As funções daqui são essas contas:

- `bfs_paths_to_fruits`: menor caminho da cabeça até cada fruta;
- `body_after_path`:     onde a cobra estaria depois de seguir um caminho;
- `flood_fill`:          quantas células dá para alcançar a partir de um ponto;
- `tail_reachable`:      dá para chegar até a própria cauda? (= "não estou preso").

Convenção: células são tuplas (linha, coluna); o corpo é uma lista com a cabeça
no índice 0 e a cauda no último índice.
"""

from __future__ import annotations

from collections import deque

from snake.env import DIRECTION_DELTAS

Cell = tuple[int, int]


def neighbors(cell: Cell, height: int, width: int):
    """Gera (direção, célula vizinha) para as vizinhas dentro do tabuleiro.

    `yield` transforma a função num gerador: ela entrega um valor por vez, como
    um `function*` do JavaScript.
    """
    r, c = cell
    for direction, (dr, dc) in DIRECTION_DELTAS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield direction, (nr, nc)


def direction_between(a: Cell, b: Cell) -> int:
    """Direção para ir da célula `a` até a vizinha `b`."""
    delta = (b[0] - a[0], b[1] - a[1])
    for direction, d in DIRECTION_DELTAS.items():
        if d == delta:
            return direction
    raise ValueError(f"{a} e {b} não são vizinhas")


def bfs_paths_to_fruits(state: dict) -> list[list[Cell]]:
    """Menor caminho da cabeça até cada fruta alcançável, do mais curto ao mais longo.

    Cada caminho é a lista de células a pisar (sem a cabeça atual; a última é a
    fruta). BFS = busca em largura: explora o mapa em "ondas" (todas as células
    a 1 passo, depois a 2, ...), por isso a 1ª vez que chega numa célula é pelo
    caminho mais curto.

    Detalhe importante: o corpo ANDA enquanto a cobra anda. Uma célula do corpo
    com índice i (cabeça = 0) fica livre depois de len(corpo) - i passos, porque
    a cauda vai saindo. Então, chegando lá no passo d, ela está livre se
    i >= len(corpo) - d. Sem isso, o agente acharia que está preso quando não está.
    """
    body = state["snake_body"]
    height, width = state["board_height"], state["board_width"]
    obstacles = set(state["obstacles"])
    fruits = set(state["fruits"])
    length = len(body)
    body_index = {cell: i for i, cell in enumerate(body)}

    head = body[0]
    parent: dict[Cell, Cell | None] = {head: None}
    queue = deque([(head, 0)])
    paths = []
    while queue:
        cell, dist = queue.popleft()
        for _, nxt in neighbors(cell, height, width):
            if nxt in parent or nxt in obstacles:
                continue
            i = body_index.get(nxt)
            if i is not None and i < length - (dist + 1):
                continue  # ainda vai estar ocupada pelo corpo quando chegarmos
            parent[nxt] = cell
            if nxt in fruits:
                paths.append(_rebuild_path(parent, nxt))
                continue  # pisar numa fruta já é comê-la: não seguimos além dela
            queue.append((nxt, dist + 1))
    return paths


def _rebuild_path(parent: dict, end: Cell) -> list[Cell]:
    """Volta pelos "pais" desde o fim até a cabeça e inverte."""
    path = []
    cell = end
    while parent[cell] is not None:
        path.append(cell)
        cell = parent[cell]
    path.reverse()
    return path


def body_after_path(body: list[Cell], path: list[Cell], ate_at_end: bool) -> list[Cell]:
    """Corpo da cobra depois de seguir `path` (comendo só na última célula, se `ate_at_end`).

    Exemplo: corpo [(4,3),(4,2),(4,1)] seguindo [(4,4),(4,5)] sem comer
    -> [(4,5),(4,4),(4,3)]: a cabeça avançou 2 e a cauda saiu 2.
    Se comer no fim, a cobra fica 1 célula maior: [(4,5),(4,4),(4,3),(4,2)].
    """
    new_length = len(body) + (1 if ate_at_end else 0)
    return (list(reversed(path)) + list(body))[:new_length]


def flood_fill(start: Cell, blocked: set[Cell], height: int, width: int) -> int:
    """Quantas células livres dá para alcançar a partir de `start` (sem contar ela).

    É o "balde de tinta" de um editor de imagem: espalha para as vizinhas até
    bater em algo bloqueado.
    """
    seen = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for _, nxt in neighbors(cell, height, width):
            if nxt not in seen and nxt not in blocked:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) - 1


def tail_reachable(body: list[Cell], obstacles: set[Cell], height: int, width: int) -> bool:
    """A cabeça consegue chegar até a cauda?

    Se consegue, a cobra nunca está presa: no pior caso ela "persegue a própria
    cauda", que vai abrindo espaço à frente. É o teste de segurança do agente BFS.
    """
    if len(body) < 2:
        return True
    head, tail = body[0], body[-1]
    blocked = obstacles | set(body[:-1])  # a cauda em si é o destino, não bloqueia
    seen = {head}
    queue = deque([head])
    while queue:
        cell = queue.popleft()
        for _, nxt in neighbors(cell, height, width):
            if nxt == tail:
                # Encostar na cauda logo no 1º passo só vale se a cobra tiver
                # corpo suficiente para a cauda sair do lugar antes (len >= 3).
                if cell != head or len(body) > 2:
                    return True
            if nxt not in seen and nxt not in blocked:
                seen.add(nxt)
                queue.append(nxt)
    return False
