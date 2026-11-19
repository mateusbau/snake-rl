"""Testes da Fase 7: checagem de segurança, solucionador de fim de jogo e paridade."""

import numpy as np

from agents.ppo import PPOAgent
from agents.search import (
    EndgameAgent,
    HunterAgent,
    SafetyAgent,
    color_imbalance,
    find_safe_meal,
    solve_endgame,
)
from snake.env import DOWN, LEFT, RIGHT, UP, GameConfig, SnakeGame

FAR_FRUITS = [(8, 9), (8, 7), (8, 5), (6, 9), (8, 3)]
# Linha única de 5 células (sem a regra de becos, que não faz sentido numa linha).
ROW = GameConfig(height=1, width=5, num_fruits=2, forbid_dead_ends=False,
                 initial_snake=((0, 2), (0, 1), (0, 0)), initial_fruits=((0, 3), (0, 4)))


class Prefers:
    """Agente de mentira com uma ordem fixa de preferência (para testar a busca)."""

    name = "stub"

    def __init__(self, order):
        self.order = order

    def reset(self, seed=None):
        pass

    def action_scores(self, game):
        scores = np.zeros(4)
        for rank, d in enumerate(self.order):
            scores[d] = 4 - rank
        return scores

    def act(self, game):
        return self.order[0]


def test_color_imbalance():
    assert color_imbalance([], 9, 10) == 0
    assert color_imbalance([(0, 0), (0, 2)], 9, 10) == 2   # duas casas "pretas"
    assert color_imbalance([(0, 0), (0, 1)], 9, 10) == 0   # uma de cada cor


def test_safety_vetoes_a_move_into_a_pocket():
    #   linha 0:  .  #  ...        A rede prefere ESQUERDA, para (1,0): um bolsão de 2 casas
    #   linha 1:  .  H  o  o       ((1,0) e (0,0)) de onde a cauda é inalcançável.
    #   linha 2:  #  .  ...        A busca veta e escolhe BAIXO.
    game = SnakeGame(seed=0)
    game.load_state(snake=[(1, 1), (1, 2), (1, 3)], direction=LEFT, fruits=FAR_FRUITS, obstacles=[(0, 1), (2, 0)])
    assert LEFT in game.safe_directions()          # não mata AGORA...
    agent = SafetyAgent(Prefers([LEFT, UP, RIGHT, DOWN]))
    assert agent.act(game) == DOWN                 # ...mas encurrala; a busca evita
    assert agent.vetoes == 1


def test_safety_follows_the_network_when_it_is_safe():
    game = SnakeGame(seed=0)
    agent = SafetyAgent(Prefers([UP, RIGHT, DOWN, LEFT]))
    assert agent.act(game) == UP and agent.vetoes == 0


def test_endgame_solver_finds_the_win():
    #   o o H * *   sem célula vazia: comer (0,3) e depois (0,4) zera
    game = SnakeGame(ROW, seed=0)
    assert solve_endgame(game, budget=1000) == [RIGHT, RIGHT]
    agent = EndgameAgent(Prefers([UP]))
    while not game.game_over:
        game.step(agent.act(game))
    assert game.end_reason == "win" and agent.solved == 1


def test_endgame_solver_proves_impossible():
    #   * H o o *   frutas nas duas pontas: comendo uma, a outra fica do outro lado do corpo
    game = SnakeGame(ROW, seed=0)
    game.load_state(snake=[(0, 1), (0, 2), (0, 3)], direction=LEFT, fruits=[(0, 0), (0, 4)])
    assert game.empty_cells() == []
    assert solve_endgame(game, budget=1000) is None


def test_ppo_scores_are_probabilities_and_respect_the_mask():
    agent = PPOAgent.new("features_space", masked=True, seed=0)
    game = SnakeGame(seed=0)
    game.load_state(snake=[(4, 2), (4, 1), (4, 0)], direction=RIGHT, fruits=FAR_FRUITS, obstacles=[(4, 3)])
    p = agent.action_scores(game)
    assert abs(p.sum() - 1) < 1e-5
    assert p[RIGHT] == 0 and p[LEFT] == 0          # obstáculo à frente; meia-volta = seguir reto


ONE_FRUIT = GameConfig(num_fruits=1, initial_fruits=((4, 5),))


def test_hunter_finds_a_simple_meal():
    game = SnakeGame(ONE_FRUIT, seed=0)  # cabeça (4,3) olhando para a direita, fruta em (4,5)
    assert find_safe_meal(game, budget=1000) == [RIGHT, RIGHT]


def test_hunter_refuses_a_fruit_inside_a_pocket():
    # Fruta em (0,0), num bolsão: só se entra por (1,0), e depois de comer a cabeça
    # fica cercada (parede, obstáculo em (0,1) e o próprio corpo). Nenhum caminho serve.
    game = SnakeGame(ONE_FRUIT, seed=0)
    game.load_state(snake=[(1, 1), (1, 2), (1, 3)], direction=LEFT, fruits=[(0, 0)], obstacles=[(0, 1), (2, 0)])
    assert find_safe_meal(game, budget=100_000) is None


def test_hunter_only_acts_after_the_trigger():
    game = SnakeGame(ONE_FRUIT, seed=0)
    agent = HunterAgent(Prefers([DOWN]), trigger=50)
    assert agent.act(game) == DOWN and agent.searches == 0   # comeu há pouco: segue a base
    game.steps_since_fruit = 60
    assert agent.act(game) == RIGHT and agent.found == 1     # faminta: caça
