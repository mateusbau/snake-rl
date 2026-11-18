"""Testes do motor do jogo (Fase 1).

Os testes numerados de 1 a 15 são exatamente os da seção 10 da especificação.
Depois deles vêm testes extras para as decisões do CLAUDE.md.

Dica de leitura: muitos testes montam um cenário com `game.load_state(...)`,
desenhado no comentário ao lado (H = cabeça, o = corpo, * = fruta, # = obstáculo).
"""

import math

import numpy as np
import pytest

from snake.checks import check_invariants
from snake.env import (
    BODY,
    DOWN,
    LEFT,
    NEIGHBORS_8,
    OBSTACLE,
    RIGHT,
    TIMEOUT,
    UP,
    WALL,
    WIN,
    GameConfig,
    SnakeGame,
)

# Frutas "longe de tudo", na linha 0, para cenários em que elas não importam.
FAR_FRUITS = [(0, 0), (0, 2), (0, 4), (0, 6), (0, 8)]


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------
def play_random_games(n_games: int, seed: int = 0, safe: bool = True):
    """Gera partidas aleatórias, devolvendo (jogo, resultado_do_passo) a cada passo.

    `yield` transforma a função num "gerador": ela entrega um valor por vez,
    como um function* com yield no JavaScript.

    safe=True: o agente sorteia só entre direções que não matam na hora, para
    as partidas durarem mais e gerarem bastante obstáculo.
    """
    rng = np.random.default_rng(seed)
    for game_seed in range(n_games):
        game = SnakeGame(seed=game_seed)
        while not game.game_over:
            options = game.safe_directions() if safe else [UP, RIGHT, DOWN, LEFT]
            if not options:
                options = [game.direction]  # sem saída: segue reto e morre
            result = game.step(int(rng.choice(options)))
            yield game, result


def chi2_critical(df: int, z: float = 3.09) -> float:
    """Valor crítico aproximado do qui-quadrado (aproximação de Wilson-Hilferty).

    z = 3.09 corresponde a 0,1% de chance de rejeitar um sorteio que é uniforme.
    """
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def chi2_statistic(counts: list[int]) -> float:
    expected = sum(counts) / len(counts)
    return sum((c - expected) ** 2 / expected for c in counts)


# ---------------------------------------------------------------------------
# Os 15 testes da seção 10 da especificação
# ---------------------------------------------------------------------------
def test_01_board_has_90_cells():
    game = SnakeGame(seed=0)
    state = game.get_state()
    assert state["board_height"] == 9 and state["board_width"] == 10
    assert state["board_height"] * state["board_width"] == 90
    # O desenho em texto também tem 9 linhas de 10 colunas.
    rows = str(game).split("\n")
    assert len(rows) == 9 and all(len(row) == 10 for row in rows)


def test_02_five_fruits_at_start():
    for seed in range(20):
        game = SnakeGame(seed=seed)
        # Sempre as mesmas 5 posições fixas, qualquer que seja a seed.
        assert game.fruits == [(2, 4), (2, 8), (4, 6), (6, 4), (6, 8)]


def test_03_snake_starts_with_3_segments():
    game = SnakeGame(seed=0)
    assert list(game.snake) == [(4, 3), (4, 2), (4, 1)]
    assert game.head == (4, 3)
    assert game.direction == RIGHT


def test_04_first_fruit_generates_obstacle():
    # Partida padrão: andando 3 vezes para a direita, a cobra come a fruta (4,6).
    game = SnakeGame(seed=0)
    game.step(RIGHT)
    game.step(RIGHT)
    assert game.obstacles_generated == 0
    result = game.step(RIGHT)
    assert result.ate_fruit and game.fruits_collected == 1
    assert game.obstacles_generated == 1 and result.new_obstacle is not None


def _eat_in_a_row(n_fruits: int, seed: int = 0):
    """Cobra andando para a direita com frutas enfileiradas à frente.

        . . . . . . . . . .
        o o H * * * * * . .   <- linha 4: come uma fruta a cada passo

    Devolve a lista com o total de obstáculos depois de cada fruta.
    """
    game = SnakeGame(seed=seed)
    game.load_state(
        snake=[(4, 3), (4, 2), (4, 1)],
        direction=RIGHT,
        fruits=[(4, 4), (4, 5), (4, 6), (4, 7), (4, 8)][:n_fruits] + FAR_FRUITS[: 5 - n_fruits],
    )
    totals = []
    for _ in range(n_fruits):
        result = game.step(RIGHT)
        assert result.ate_fruit
        totals.append(game.obstacles_generated)
    return totals


def test_05_second_fruit_does_not_add_obstacle():
    for seed in range(50):
        totals = _eat_in_a_row(2, seed)
        assert totals == [1, 1]


def test_06_third_fruit_adds_another_obstacle():
    # Confere a tabela da seção 3.1 até 5 frutas: 1→1, 2→1, 3→2, 4→2, 5→3.
    for seed in range(50):
        assert _eat_in_a_row(5, seed) == [1, 1, 2, 2, 3]


def test_07_obstacles_are_static():
    # Em partidas longas, a lista de obstáculos só cresce no fim; os antigos
    # nunca mudam de lugar nem somem.
    last_game, old = None, []
    for game, _ in play_random_games(200):
        if game is not last_game:  # começou uma partida nova
            last_game, old = game, []
        assert game.obstacles[: len(old)] == old
        old = list(game.obstacles)


def test_08_obstacles_never_adjacent():
    # check_invariants confere as 8 vizinhas de cada obstáculo, a cada passo.
    max_obstacles = 0
    for game, _ in play_random_games(300):
        check_invariants(game)
        max_obstacles = max(max_obstacles, len(game.obstacles))
    assert max_obstacles >= 8  # garante que o teste viu tabuleiros cheios de obstáculos


def test_09_obstacles_never_on_fruit_or_snake():
    for game, result in play_random_games(300):
        cell = result.new_obstacle
        if cell is not None:
            # O obstáculo é o último a nascer no passo, então o estado atual
            # é exatamente o do momento do sorteio.
            assert cell not in game.fruits
            assert cell not in game.snake
        # E nunca há sobreposição depois disso (cobra/fruta/obstáculo).
        check_invariants(game)


def test_10_safety_zone_respected():
    # (a) A zona acompanha a direção e corta o que sai do mapa.
    game = SnakeGame(seed=0)
    game.load_state(snake=[(4, 3), (4, 2), (4, 1)], direction=RIGHT, fruits=FAR_FRUITS)
    assert game.safety_zone() == [(4, 4), (4, 5), (4, 6)]
    game.load_state(snake=[(4, 3), (5, 3), (6, 3)], direction=UP, fruits=FAR_FRUITS)
    assert game.safety_zone() == [(3, 3), (2, 3), (1, 3)]
    game.load_state(snake=[(4, 8), (4, 7), (4, 6)], direction=RIGHT, fruits=FAR_FRUITS)
    assert game.safety_zone() == [(4, 9)]  # só 1 célula cabe no mapa

    # (b) Essas células somem da lista de posições válidas (mas não as diagonais).
    game.load_state(snake=[(4, 3), (4, 2), (4, 1)], direction=RIGHT, fruits=FAR_FRUITS)
    eligible = game.eligible_obstacle_cells()
    assert not {(4, 4), (4, 5), (4, 6)} & set(eligible)
    assert {(3, 4), (5, 4), (4, 7)} <= set(eligible)  # sem cone, profundidade exata 3

    # (c) Nas partidas, nenhum obstáculo nasce na zona do momento em que nasce.
    for game, result in play_random_games(300):
        if result.new_obstacle is not None:
            assert result.new_obstacle not in game.safety_zone()


def test_11_obstacles_can_spawn_next_to_body_and_fruits():
    game = SnakeGame(seed=0)
    game.load_state(snake=[(4, 3), (4, 2), (4, 1)], direction=RIGHT, fruits=FAR_FRUITS)
    eligible = set(game.eligible_obstacle_cells())
    assert (5, 2) in eligible and (3, 1) in eligible   # colados no corpo
    assert (1, 4) in eligible and (0, 3) in eligible   # colados em frutas

    # E isso de fato acontece nas partidas.
    next_to_body = next_to_fruit = 0
    for game, result in play_random_games(300):
        cell = result.new_obstacle
        if cell is None:
            continue
        around = {(cell[0] + dr, cell[1] + dc) for dr, dc in NEIGHBORS_8}
        next_to_body += bool(around & set(game.snake))
        next_to_fruit += bool(around & set(game.fruits))
    assert next_to_body > 0 and next_to_fruit > 0


def test_12_obstacle_position_is_uniform():
    # Repete o MESMO sorteio 20.000 vezes e confere se todas as células válidas
    # saem com frequência parecida (teste qui-quadrado).
    game = SnakeGame(seed=123)
    scenario = dict(snake=[(4, 3), (4, 2), (4, 1)], direction=RIGHT, fruits=FAR_FRUITS, obstacles=[(7, 7)])
    game.load_state(**scenario)
    eligible = game.eligible_obstacle_cells()
    counts = {cell: 0 for cell in eligible}
    for _ in range(20_000):
        game.load_state(**scenario)
        counts[game.try_spawn_obstacle()] += 1
    assert all(n > 0 for n in counts.values())  # nenhuma célula válida é esquecida
    assert chi2_statistic(list(counts.values())) < chi2_critical(len(counts) - 1)


def test_13_no_valid_position_for_obstacle():
    #   . . . . . . . . . .
    #   . o o H * . . . . .     <- linha 1: a cobra vai comer a fruta (1,4)
    # Primeiro enchemos o mapa de obstáculos até não caber mais nenhum (sorteando
    # com as regras normais). Depois a cobra come a 1ª fruta — marco ímpar — e a
    # tentativa tem que ser cancelada sem erro.
    fruits = [(1, 4), (0, 0), (8, 9), (8, 0), (0, 9)]
    scenario = dict(snake=[(1, 3), (1, 2), (1, 1)], direction=RIGHT, fruits=fruits)
    game = SnakeGame(seed=0)
    game.load_state(**scenario)
    while game.try_spawn_obstacle() is not None:
        pass
    saturated = list(game.obstacles)
    game.load_state(**scenario, obstacles=saturated)
    assert game.eligible_obstacle_cells() == []

    # Ao comer, a cobra cresce (a cauda não sai), então nenhuma célula nova fica livre.
    result = game.step(RIGHT)
    assert result.ate_fruit and game.fruits_collected == 1
    assert result.new_obstacle is None
    assert game.obstacles == saturated     # nenhum obstáculo novo
    assert len(saturated) <= 25            # nunca passa de 25
    assert not game.game_over              # o jogo continua normalmente
    check_invariants(game)


def test_14_fruit_replenishment():
    # (a) Depois de comer, volta a ter 5 frutas, numa célula que estava vazia.
    for seed in range(50):
        game = SnakeGame(seed=seed)
        game.step(RIGHT)
        game.step(RIGHT)
        empty_before = set(game.empty_cells()) - {(4, 6)}
        result = game.step(RIGHT)
        assert len(game.fruits) == 5
        assert result.new_fruit in empty_before

    # (b) Reposição uniforme entre as células vazias (qui-quadrado).
    game = SnakeGame(seed=7)
    scenario = dict(snake=[(4, 3), (4, 2), (4, 1)], direction=RIGHT, fruits=[(4, 4)] + FAR_FRUITS[:4])
    counts: dict = {}
    for _ in range(20_000):
        game.load_state(**scenario)
        cell = game.step(RIGHT).new_fruit
        counts[cell] = counts.get(cell, 0) + 1
    assert len(counts) == 90 - 3 - 1 - 4  # todas as células vazias apareceram
    assert chi2_statistic(list(counts.values())) < chi2_critical(len(counts) - 1)

    # (c) Sem célula vazia para repor: não nasce fruta nova (sem erro), mas a
    #     partida CONTINUA enquanto houver fruta no tabuleiro. Tabuleiro 1x5:
    #     o o H * *   -> come (0,3): sobra a fruta (0,4) e nenhuma célula vazia.
    # (forbid_dead_ends=False: numa linha só, as pontas sempre seriam "becos".)
    tiny = GameConfig(height=1, width=5, num_fruits=2, forbid_dead_ends=False,
                      initial_snake=((0, 2), (0, 1), (0, 0)), initial_fruits=((0, 3), (0, 4)))
    game = SnakeGame(tiny, seed=0)
    result = game.step(RIGHT)
    assert result.ate_fruit and result.new_fruit is None
    assert not result.game_over and game.fruits == [(0, 4)] and game.empty_cells() == []
    check_invariants(game)
    # (d) Come a última fruta: agora só há cobra -> vitória ("zerou").
    result = game.step(RIGHT)
    assert result.game_over and result.end_reason == WIN and game.is_won()


# ---------------------------------------------------------------------------
# Vitória ("zerar"): cobra + obstáculos ocupam TODAS as células (sem fruta e sem vazio)
# ---------------------------------------------------------------------------
TINY_1x5 = GameConfig(height=1, width=5, num_fruits=1,
                      initial_snake=((0, 1), (0, 0)), initial_fruits=((0, 2),))


def test_win_when_only_snake_and_obstacles_remain():
    #   o o H * #   -> come (0,3): cobra 4 + obstáculo 1 = as 5 células
    game = SnakeGame(TINY_1x5, seed=0)
    game.load_state(snake=[(0, 2), (0, 1), (0, 0)], direction=RIGHT, fruits=[(0, 3)], obstacles=[(0, 4)])
    assert not game.is_won()
    result = game.step(RIGHT)
    assert result.game_over and result.end_reason == WIN
    assert game.is_won() and game.fruits == [] and result.new_obstacle is None


def test_not_won_with_fruit_left_and_no_empty_cell():
    #   o H * * #   cobra 2 + 2 frutas + obstáculo 1 = 5 células, nenhuma vazia
    game = SnakeGame(TINY_1x5, seed=0)
    game.load_state(snake=[(0, 1), (0, 0)], direction=RIGHT, fruits=[(0, 2), (0, 3)], obstacles=[(0, 4)])
    assert game.empty_cells() == [] and not game.is_won()
    result = game.step(RIGHT)  # come (0,2); ainda sobra (0,3)
    assert result.ate_fruit and not result.game_over and not game.is_won()
    assert game.fruits == [(0, 3)]
    result = game.step(RIGHT)  # come a última
    assert result.end_reason == WIN


def test_not_won_with_an_empty_cell():
    #   o H . . *   há células vazias -> não é vitória (nem antes nem depois de andar)
    game = SnakeGame(TINY_1x5, seed=0)
    game.load_state(snake=[(0, 1), (0, 0)], direction=RIGHT, fruits=[(0, 4)])
    assert not game.is_won()
    result = game.step(RIGHT)
    assert not result.game_over and not game.is_won()
    # E um tabuleiro de verdade no começo também não.
    assert not SnakeGame(seed=0).is_won()


def test_15_all_defeat_conditions():
    # Parede, nas 4 bordas.
    walls = [
        ([(0, 5), (1, 5), (2, 5)], UP),
        ([(8, 5), (7, 5), (6, 5)], DOWN),
        ([(4, 0), (4, 1), (4, 2)], LEFT),
        ([(4, 9), (4, 8), (4, 7)], RIGHT),
    ]
    for snake, direction in walls:
        game = SnakeGame(seed=0)
        game.load_state(snake=snake, direction=direction, fruits=[(8, 0), (8, 9), (0, 0), (0, 9), (2, 2)])
        result = game.step(direction)
        assert result.game_over and result.end_reason == WALL

    # Obstáculo:  o o H #
    game = SnakeGame(seed=0)
    game.load_state(snake=[(4, 3), (4, 2), (4, 1)], direction=RIGHT, fruits=FAR_FRUITS, obstacles=[(4, 4)])
    assert game.step(RIGHT).end_reason == OBSTACLE

    # Próprio corpo: a cobra está em "U" e vira para baixo, em cima do corpo.
    #     H o        linha 4
    #   o o o        linha 5   (cauda em (5,3))
    game = SnakeGame(seed=0)
    game.load_state(snake=[(4, 4), (4, 5), (5, 5), (5, 4), (5, 3)], direction=LEFT, fruits=FAR_FRUITS)
    assert game.step(DOWN).end_reason == BODY

    # Depois de perder, não dá para continuar jogando sem reset().
    with pytest.raises(RuntimeError):
        game.step(LEFT)


# ---------------------------------------------------------------------------
# Testes extras: decisões do CLAUDE.md
# ---------------------------------------------------------------------------
def test_entering_tail_cell_is_allowed():
    #     H o        linha 4
    #     o o        linha 5   (cauda em (5,4), sai do lugar neste passo)
    game = SnakeGame(seed=0)
    game.load_state(snake=[(4, 4), (4, 5), (5, 5), (5, 4)], direction=LEFT, fruits=FAR_FRUITS)
    result = game.step(DOWN)
    assert not result.game_over
    assert game.head == (5, 4)


def test_reverse_command_is_ignored():
    game = SnakeGame(seed=0)
    game.step(LEFT)  # andando para a direita, LEFT seria meia-volta
    assert game.direction == RIGHT and game.head == (4, 4)


def test_snake_grows_one_cell_per_fruit():
    game = SnakeGame(seed=0)
    for _ in range(3):
        game.step(RIGHT)
    assert len(game.snake) == 4 and game.fruits_collected == 1


def test_timeout_after_steps_without_fruit():
    config = GameConfig(max_steps_without_fruit=5)
    game = SnakeGame(config, seed=0)
    game.load_state(snake=[(8, 3), (8, 2), (8, 1)], direction=RIGHT, fruits=FAR_FRUITS)
    for _ in range(4):
        assert not game.step(RIGHT).game_over
    assert game.step(RIGHT).end_reason == TIMEOUT


def test_same_seed_same_game():
    # Reprodutibilidade: mesma seed + mesmas ações = mesma partida, passo a passo.
    def run(seed):
        game = SnakeGame(seed=seed)
        rng = np.random.default_rng(99)
        history = []
        while not game.game_over:
            options = game.safe_directions() or [game.direction]
            game.step(int(rng.choice(options)))
            history.append(str(game))
        return history

    assert run(5) == run(5)
    assert run(5) != run(6)


# ---------------------------------------------------------------------------
# Seção 4.5: proibição de becos sem saída (regra nossa, não existe no Google)
# ---------------------------------------------------------------------------
# Estes testes usam funções auxiliares PRÓPRIAS (abaixo), escritas sem olhar a
# implementação do jogo: assim um erro no jogo não "se esconde" num erro igual no teste.

def free_cells(game, obstacles: set) -> set:
    """Só bordas e obstáculos bloqueiam. Cobra e frutas contam como livres."""
    return {(r, c) for r in range(game.config.height) for c in range(game.config.width)} - obstacles


def has_dead_end(game, obstacles: set | None = None) -> bool:
    free = free_cells(game, set(game.obstacles) if obstacles is None else obstacles)
    for r, c in free:
        exits = sum(n in free for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        if exits < 2:
            return True
    return False


def all_free_connected(game) -> bool:
    """BFS (busca em largura) a partir de uma célula livre: alcança todas?"""
    free = free_cells(game, set(game.obstacles))
    start = min(free)
    seen, frontier = {start}, [start]
    while frontier:
        r, c = frontier.pop()
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nxt in free and nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return len(seen) == len(free)


def creates_dead_end_independent(game, cell) -> bool:
    """Põe um obstáculo de mentira em `cell` e procura becos no mapa INTEIRO."""
    return has_dead_end(game, set(game.obstacles) | {cell})


def old_rules_eligible(game) -> set:
    """Células válidas pelas regras 4.1 a 4.4 (sem a 4.5), calculadas à mão."""
    obstacles = set(game.obstacles)
    occupied = set(game.snake) | set(game.fruits) | obstacles
    zone = set(game.safety_zone())
    result = set()
    for r in range(game.config.height):
        for c in range(game.config.width):
            near = any((r + dr, c + dc) in obstacles for dr, dc in NEIGHBORS_8)
            if (r, c) not in occupied and (r, c) not in zone and not near:
                result.add((r, c))
    return result


def test_16_cells_next_to_corners_are_invalid():
    # Tabuleiro sem obstáculos. Um obstáculo em (0,1) deixaria o canto (0,0) com
    # uma saída só, (1,0): um beco. O mesmo vale para os 4 cantos.
    #   x # . .
    #   . . . .
    game = SnakeGame(seed=0)
    eligible = set(game.eligible_obstacle_cells())
    next_to_corners = [(0, 1), (1, 0), (0, 8), (1, 9), (7, 0), (8, 1), (8, 8), (7, 9)]
    for cell in next_to_corners:
        assert game.creates_dead_end(cell), cell
        assert cell not in eligible, cell
    # O próprio canto pode receber obstáculo (não deixa ninguém com 1 saída).
    assert not game.creates_dead_end((0, 0)) and (0, 0) in eligible


def test_17_gap_against_wall_is_invalid():
    # Obstáculo em (0,0). Um novo em (0,2) prenderia (0,1) entre a parede e os dois:
    #   # x # .
    #   . . . .
    game = SnakeGame(seed=0)
    game.load_state(snake=[(4, 3), (4, 2), (4, 1)], direction=RIGHT,
                    fruits=[(2, 4), (2, 8), (4, 6), (6, 4), (6, 8)], obstacles=[(0, 0)])
    assert game.creates_dead_end((0, 2))
    assert (0, 2) not in game.eligible_obstacle_cells()
    # Um pouco mais longe já pode: (0,1) e (0,2) ficam com 2 saídas cada.
    assert (0, 3) in game.eligible_obstacle_cells()


def test_18_corridor_in_the_middle_is_valid():
    # Obstáculo em (4,4). Um novo em (4,6) deixa (4,5) com saídas para cima e para baixo:
    #   . . . . .
    #   . # . # .     <- corredor, não beco
    #   . . . . .
    game = SnakeGame(seed=0)
    game.load_state(snake=[(8, 3), (8, 2), (8, 1)], direction=RIGHT, fruits=FAR_FRUITS, obstacles=[(4, 4)])
    assert not game.creates_dead_end((4, 6))
    assert (4, 6) in game.eligible_obstacle_cells()


def test_19_eligible_list_excludes_dead_end_cells():
    # Em estados reais de partidas, a lista de válidas tem que ser exatamente:
    # (válidas pelas regras 4.1-4.4) menos (as que criariam beco), com o beco
    # conferido pela busca independente no mapa inteiro.
    checked = 0
    for game, _ in play_random_games(150, seed=1):
        if game.game_over or game.steps % 5:
            continue
        expected = {cell for cell in old_rules_eligible(game) if not creates_dead_end_independent(game, cell)}
        assert set(game.eligible_obstacle_cells()) == expected
        checked += 1
    assert checked > 1000


def test_20_no_dead_ends_and_free_cells_connected():
    # Propriedade: depois de CADA passo de milhares de partidas, nenhuma célula livre
    # tem menos de 2 vizinhas livres, e todas as livres continuam conectadas.
    games = 0
    for game, result in play_random_games(2000, seed=2):
        assert not has_dead_end(game)
        assert all_free_connected(game)
        games += result.game_over
    assert games == 2000


# ---------------------------------------------------------------------------
# Tabuleiro de treino sem obstáculos (Fase 4, pedido no CLAUDE.md: "5×5 sem obstáculos")
# ---------------------------------------------------------------------------
MINI = GameConfig(
    height=5, width=5, num_fruits=1,
    initial_snake=((2, 2), (2, 1), (2, 0)), initial_direction=RIGHT,
    initial_fruits=((2, 4),), max_steps_without_fruit=50,
    spawn_obstacles=False,
)


def test_official_game_still_spawns_obstacles():
    # O padrão continua sendo o jogo oficial: 1 obstáculo a cada quantidade ímpar.
    assert GameConfig().spawn_obstacles is True
    game = SnakeGame(seed=0)
    game.step(None)          # (4,4)
    game.step(None)          # (4,5)
    result = game.step(None)  # (4,6) = fruta inicial -> 1ª fruta (ímpar)
    assert result.ate_fruit and result.new_obstacle is not None


def test_spawn_obstacles_false_never_creates_obstacles():
    rng = np.random.default_rng(0)
    fruits_total = 0
    for seed in range(300):
        game = SnakeGame(MINI, seed=seed)
        while not game.game_over:
            game.step(int(rng.choice(game.safe_directions() or [game.direction])))
            check_invariants(game)
        assert game.obstacles == [] and game.obstacles_generated == 0
        fruits_total += game.fruits_collected
    assert fruits_total > 300  # houve várias frutas ímpares (e nenhum obstáculo)
