"""Testes dos agentes sem ML e da avaliação padronizada (Fase 3).

Cenários desenhados no comentário ao lado (H = cabeça, o = corpo, * = fruta, # = obstáculo).
Linhas e colunas começam em 0.
"""

import pytest

from agents.grid import (
    bfs_paths_to_fruits,
    body_after_path,
    direction_between,
    flood_fill,
    tail_reachable,
)
from agents.simple import AGENTS, BfsAgent, GreedyAgent, SafeRandomAgent
from snake.env import DOWN, LEFT, RIGHT, UP, SnakeGame
from snake.evaluation import EVAL_GAMES, eval_seeds, evaluate, play_game, summarize

FAR_FRUITS = [(0, 0), (0, 2), (0, 4), (0, 6), (0, 8)]


def scenario(snake, direction, fruits, obstacles=()):
    game = SnakeGame(seed=0)
    game.load_state(snake=snake, direction=direction, fruits=fruits, obstacles=obstacles)
    return game


# ---------------------------------------------------------------------------
# Sementes oficiais
# ---------------------------------------------------------------------------
def test_eval_seeds_are_fixed_and_unique():
    seeds = eval_seeds()
    assert len(seeds) == EVAL_GAMES == 1000
    assert len(set(seeds)) == 1000
    assert seeds == eval_seeds()                 # sempre as mesmas
    assert eval_seeds(10) == seeds[:10]          # um pedaço é o começo da prova oficial
    assert all(isinstance(s, int) for s in seeds)


# ---------------------------------------------------------------------------
# Ferramentas de mapa (agents/grid.py)
# ---------------------------------------------------------------------------
def test_body_after_path():
    body = [(4, 3), (4, 2), (4, 1)]
    assert body_after_path(body, [(4, 4), (4, 5)], ate_at_end=False) == [(4, 5), (4, 4), (4, 3)]
    assert body_after_path(body, [(4, 4), (4, 5)], ate_at_end=True) == [(4, 5), (4, 4), (4, 3), (4, 2)]
    # Caminho mais longo que a cobra: o corpo inteiro vira parte do caminho.
    assert body_after_path(body, [(4, 4), (4, 5), (4, 6), (4, 7)], False) == [(4, 7), (4, 6), (4, 5)]


def test_direction_between():
    assert direction_between((4, 4), (3, 4)) == UP
    assert direction_between((4, 4), (4, 5)) == RIGHT
    assert direction_between((4, 4), (5, 4)) == DOWN
    assert direction_between((4, 4), (4, 3)) == LEFT
    with pytest.raises(ValueError):
        direction_between((4, 4), (6, 4))


def test_flood_fill_counts_reachable_cells():
    # Tabuleiro vazio 9x10: a partir de um canto, alcança as outras 89 células.
    assert flood_fill((0, 0), set(), 9, 10) == 89
    # Um "muro" na coluna 1 isola a coluna 0 (9 células, 8 além do início).
    wall = {(r, 1) for r in range(9)}
    assert flood_fill((0, 0), wall, 9, 10) == 8


def test_bfs_goes_around_obstacle():
    # Linha 4:  . . . o o H # * . .   -> a fruta está a 2 colunas, mas há um obstáculo no meio
    game = scenario([(4, 5), (4, 4), (4, 3)], RIGHT, [(4, 7), *FAR_FRUITS[:4]], obstacles=[(4, 6)])
    paths = bfs_paths_to_fruits(game.get_state())
    nearest = paths[0]
    assert nearest[-1] == (4, 7)
    assert (4, 6) not in nearest
    assert len(nearest) == 4  # contorna: sobe/desce, anda 2, volta


def test_bfs_knows_the_tail_moves_away():
    # Cobra em "U" andando para a esquerda; a fruta está logo acima da cauda.
    # O caminho passa pela casa onde HOJE está a cauda, que já terá saído quando
    # a cabeça chegar.
    #          col 0  col 1
    #   linha 2:  *     .
    #   linha 3:  o     o      (3,0) = cauda
    #   linha 4:  H     o      (4,0) = cabeça
    snake = [(4, 0), (4, 1), (3, 1), (3, 0)]
    game = scenario(snake, LEFT, [(2, 0), *FAR_FRUITS[1:]])
    paths = bfs_paths_to_fruits(game.get_state())
    # Caminho para (2,0): sobe por (3,0), que é a cauda (sai no 1º passo) -> 2 passos.
    assert [(3, 0), (2, 0)] in paths


def test_tail_reachable_detects_trap():
    # Cabeça no canto, cercada pelo próprio corpo e por um obstáculo:
    #   linha 0:  H o . .
    #   linha 1:  # o o o      (1,3) = cauda
    trapped = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)]
    assert not tail_reachable(trapped, {(1, 0)}, 9, 10)
    assert tail_reachable(trapped, set(), 9, 10)  # sem o obstáculo, desce por (1,0)


# ---------------------------------------------------------------------------
# Agentes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(AGENTS))
def test_agents_return_valid_actions_and_are_reproducible(name):
    seeds = eval_seeds(3)
    first = evaluate(AGENTS[name](), seeds)
    second = evaluate(AGENTS[name](), seeds)
    assert first == second  # mesma semente -> mesma partida, ação por ação
    for game_result in first:
        assert all(a in (UP, RIGHT, DOWN, LEFT) for a in game_result["actions"])


@pytest.mark.parametrize("name", list(AGENTS))
def test_agents_do_not_change_the_game(name):
    """O agente só consulta: o estado antes e depois de act() é o mesmo."""
    game = SnakeGame(seed=eval_seeds(1)[0])
    agent = AGENTS[name]()
    agent.reset(1)
    for _ in range(30):
        if game.game_over:
            break
        before = game.get_state()
        rng_before = game.rng.bit_generator.state
        action = agent.act(game)
        assert game.get_state() == before
        assert game.rng.bit_generator.state == rng_before  # não espiou nem gastou sorteios do jogo
        game.step(action)


def test_greedy_goes_to_nearest_fruit():
    # Fruta 2 casas acima da cabeça; o resto longe -> sobe.
    game = scenario([(4, 3), (4, 2), (4, 1)], RIGHT, [(2, 3), (8, 9), (8, 7), (0, 9), (8, 0)])
    assert GreedyAgent().act(game) == UP


def test_greedy_avoids_immediate_death():
    # A fruta está à direita, mas há um obstáculo colado à frente: o guloso desvia.
    game = scenario([(4, 3), (4, 2), (4, 1)], RIGHT, [(4, 6), *FAR_FRUITS[:4]], obstacles=[(4, 4)])
    assert GreedyAgent().act(game) in (UP, DOWN)


@pytest.mark.parametrize("agent_cls", [SafeRandomAgent, GreedyAgent, BfsAgent])
def test_smart_agents_never_choose_immediate_death_when_avoidable(agent_cls):
    for seed in eval_seeds(20):
        game = SnakeGame(seed=seed)
        agent = agent_cls()
        agent.reset(seed)
        while not game.game_over:
            safe = game.safe_directions()
            action = agent.act(game)
            if safe:
                assert action in safe, f"seed {seed}: escolheu {action}, seguras {safe}\n{game}"
            game.step(action)


def test_bfs_refuses_fruit_that_traps_it():
    # A fruta mais próxima está num bolsão de 3 casas no canto superior direito,
    # fechado por 2 obstáculos e pelo próprio corpo. Comer lá = ficar preso.
    #   col:      2 3 4 5 6 7 8 9
    #   linha 0:  . . . . . # . *     bolsão = (0,8), (0,9), (1,9)
    #   linha 1:  o o o o o o H .     cabeça (1,8) andando para a direita
    #   linha 2:  . . . . . . . #
    snake = [(1, 8), (1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2)]
    obstacles = [(2, 9), (0, 7)]
    fruits = [(0, 9), (8, 0), (8, 2), (8, 4), (8, 6)]
    game = scenario(snake, RIGHT, fruits, obstacles)
    state = game.get_state()
    path = bfs_paths_to_fruits(state)[0]
    assert path[-1] == (0, 9)
    after = body_after_path(state["snake_body"], path, ate_at_end=True)
    assert not tail_reachable(after, set(obstacles), 9, 10)  # comer = ficar preso
    action = BfsAgent().act(game)
    assert action == DOWN  # sai por baixo, em direção às outras frutas


def test_bfs_beats_greedy_on_a_small_sample():
    seeds = eval_seeds(20)
    greedy = summarize(evaluate(GreedyAgent(), seeds))
    bfs = summarize(evaluate(BfsAgent(), seeds))
    assert bfs["fruits_mean"] > greedy["fruits_mean"]


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def test_play_game_result_matches_replay():
    """O resultado de play_game reproduz exatamente via replay (seed + ações)."""
    from snake import replay as rp

    result = play_game(GreedyAgent(), eval_seeds(1)[0])
    replay = rp.Replay(seed=result["seed"], actions=result["actions"], player="greedy")
    game = rp.simulate(replay)
    assert game.fruits_collected == result["fruits"]
    assert game.steps == result["steps"]
    assert game.end_reason == result["end_reason"]


def test_summarize_metrics():
    results = [
        {"fruits": 2, "board_filled": 0.1, "steps": 10, "end_reason": "wall"},
        {"fruits": 4, "board_filled": 0.2, "steps": 30, "end_reason": "win"},
    ]
    s = summarize(results)
    assert s["games"] == 2
    assert s["fruits_mean"] == 3
    assert s["fruits_median"] == 3
    assert s["fruits_best"] == 4
    assert s["win_rate"] == 0.5
    assert s["end_reasons"] == {"wall": 0.5, "win": 0.5}
