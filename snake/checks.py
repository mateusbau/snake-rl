"""Checagens de "invariantes": regras que precisam valer em TODO momento do jogo.

Usadas pelos testes e pelo script que roda milhares de partidas aleatórias.
Se alguma falhar, levanta AssertionError com uma mensagem explicando o quê.
(`assert condição, "mensagem"` é como um `console.assert`, mas que PARA o programa.)
"""

from __future__ import annotations

from functools import lru_cache

from snake.env import NEIGHBORS_4, NEIGHBORS_8, WIN, SnakeGame


def check_invariants(game: SnakeGame) -> None:
    cfg = game.config
    snake = list(game.snake)

    # Tudo dentro do tabuleiro (quando a cobra bate na parede, ela não chega a
    # se mover, então nem nesse caso algo fica fora do mapa).
    for r, c in snake + game.fruits + game.obstacles:
        assert 0 <= r < cfg.height and 0 <= c < cfg.width, f"célula fora do mapa: {(r, c)}"

    # Nenhuma célula com dois elementos (seção 2.1).
    all_cells = snake + game.fruits + game.obstacles
    assert len(set(all_cells)) == len(all_cells), "duas coisas na mesma célula"

    # A cobra é contínua: cada segmento encosta (sem diagonal) no seguinte.
    for (r1, c1), (r2, c2) in zip(snake, snake[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "cobra quebrada"

    # Comprimento = inicial + crescimento por fruta (menos o que ainda vai crescer).
    expected_len = len(cfg.initial_snake) + cfg.growth_per_fruit * game.fruits_collected - game.pending_growth
    assert len(snake) == expected_len, f"comprimento {len(snake)} != {expected_len}"

    # Sempre 5 frutas, enquanto houver espaço (só falta fruta quando o jogo foi zerado).
    if game.end_reason != WIN:
        assert len(game.fruits) == cfg.num_fruits, f"{len(game.fruits)} frutas no mapa"

    # Obstáculos: no máximo 1 por marco ímpar (1, 3, 5, ...), descontando os que
    # já vieram prontos num cenário montado com load_state.
    attempts = (game.fruits_collected + 1) // 2
    assert game.obstacles_generated == len(game.obstacles), "contador de obstáculos errado"
    assert game.obstacles_generated - game.obstacles_at_start <= attempts, "obstáculos demais"

    # Dois obstáculos nunca encostam, nem na diagonal (seção 4.2).
    obstacles = set(game.obstacles)
    for r, c in game.obstacles:
        for dr, dc in NEIGHBORS_8:
            assert (r + dr, c + dc) not in obstacles, f"obstáculos vizinhos em {(r, c)}"

    # Seção 4.5: nenhuma célula livre com menos de 2 saídas, e todas conectadas.
    if cfg.forbid_dead_ends:
        _check_free_cells(cfg.height, cfg.width, tuple(sorted(obstacles)))


# @lru_cache guarda o resultado de chamadas anteriores ("memoização"). Como becos e
# conectividade só dependem dos obstáculos — que mudam raramente —, a conta pesada
# só roda de novo quando nasce um obstáculo. Se a checagem falhar, ela levanta erro
# e nada é guardado.
@lru_cache(maxsize=4096)
def _check_free_cells(height: int, width: int, obstacles: tuple) -> None:
    """Só paredes e obstáculos bloqueiam; cobra e frutas contam como livres."""
    free = {(r, c) for r in range(height) for c in range(width)} - set(obstacles)
    for r, c in free:
        exits = sum((r + dr, c + dc) in free for dr, dc in NEIGHBORS_4)
        assert exits >= 2, f"beco sem saída em {(r, c)}"
    # BFS: começa numa célula livre e vai "pintando" as vizinhas alcançáveis.
    start = min(free)
    seen, frontier = {start}, [start]
    while frontier:
        r, c = frontier.pop()
        for dr, dc in NEIGHBORS_4:
            n = (r + dr, c + dc)
            if n in free and n not in seen:
                seen.add(n)
                frontier.append(n)
    assert len(seen) == len(free), "células livres isoladas"
