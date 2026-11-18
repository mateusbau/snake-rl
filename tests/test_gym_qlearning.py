"""Testes da interface Gymnasium e do Q-learning com tabela (Fase 4)."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from agents.qlearning import QLearningAgent
from snake.boards import BOARDS
from snake.env import DOWN, LEFT, RIGHT, UP, SnakeGame
from snake.evaluation import eval_seeds, evaluate, play_game
from snake.gym_env import FEATURE_NAMES, Rewards, SnakeEnv, feature_observation, grid_observation

FAR_FRUITS = [(0, 0), (0, 2), (0, 4), (0, 6), (0, 8)]


# ---------------------------------------------------------------------------
# Ambiente Gymnasium
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("obs_type", ["grid", "features"])
@pytest.mark.parametrize("board", ["mini", "full"])
def test_env_passes_gymnasium_checker(obs_type, board):
    # O verificador oficial do Gymnasium confere espaços, tipos, reset/step e seeds.
    check_env(SnakeEnv(BOARDS[board], obs_type), skip_render_check=True)


def test_env_with_seed_is_the_same_game_as_the_engine():
    """reset(seed=s) + as mesmas ações = a mesma partida do motor e do evaluate."""
    seed = eval_seeds(1)[0]
    actions = [RIGHT, RIGHT, UP, UP, LEFT, DOWN, RIGHT, RIGHT, RIGHT]
    env = SnakeEnv()
    env.reset(seed=seed)
    game = SnakeGame(seed=seed)
    for a in actions:
        if game.game_over:
            break
        env.step(a)
        game.step(a)
    assert env.game.get_state() == game.get_state()


def test_rewards_fruit_death_and_timeout():
    env = SnakeEnv(BOARDS["mini"], "features")
    env.reset(seed=0)
    # Mini: cobra (2,2) olhando para a direita, fruta inicial em (2,4).
    _, r1, term, trunc, _ = env.step(RIGHT)  # (2,3)
    assert (r1, term, trunc) == (0.0, False, False)
    _, r2, term, trunc, info = env.step(RIGHT)  # (2,4) = fruta
    assert r2 == Rewards().fruit and info["fruits"] == 1 and not term
    # Parede: segue reto até sair do tabuleiro.
    _, r3, term, trunc, info = env.step(RIGHT)
    assert r3 == Rewards().death and term and not trunc and info["end_reason"] == "wall"


def test_timeout_is_truncation_not_termination():
    env = SnakeEnv(BOARDS["mini"], "grid")
    env.reset(seed=0)
    env.game.load_state(snake=[(1, 1), (1, 2), (2, 2), (2, 1)], direction=LEFT, fruits=[(4, 4)])
    # Anda em círculo (quadrado 2×2 com a cabeça perseguindo a cauda) até o limite.
    cycle = [DOWN, RIGHT, UP, LEFT]
    for i in range(BOARDS["mini"].max_steps_without_fruit):
        _, reward, terminated, truncated, info = env.step(cycle[i % 4])
    assert truncated and not terminated
    assert info["end_reason"] == "timeout" and reward == Rewards().timeout


def test_grid_observation_channels():
    game = SnakeGame(BOARDS["mini"], seed=0)
    obs = grid_observation(game)
    assert obs.shape == (4, 5, 5) and obs.dtype == np.uint8
    assert obs[0][2, 2] == 1 and obs[0].sum() == 1          # cabeça
    assert obs[1][2, 1] == obs[1][2, 0] == 1 and obs[1].sum() == 2  # corpo
    assert obs[2][2, 4] == 1 and obs[2].sum() == 1          # fruta
    assert obs[3].sum() == 0                                 # sem obstáculos


def test_feature_observation_example():
    # Cobra andando para a direita, obstáculo colado à frente, fruta acima e à direita.
    #   linha 2:  . . . * .
    #   linha 4:  o o H # .
    game = SnakeGame(seed=0)
    game.load_state(snake=[(4, 2), (4, 1), (4, 0)], direction=RIGHT,
                    fruits=[(2, 3), (8, 9), (8, 7), (0, 9), (8, 5)], obstacles=[(4, 3)])
    f = dict(zip(FEATURE_NAMES, feature_observation(game)))
    assert f["danger_straight"] == 1 and f["danger_right"] == 0 and f["danger_left"] == 0
    assert f["dir_right"] == 1 and f["dir_up"] == f["dir_down"] == f["dir_left"] == 0
    assert f["fruit_up"] == 1 and f["fruit_right"] == 1
    assert f["fruit_down"] == 0 and f["fruit_left"] == 0


def test_action_masks_match_safe_directions():
    env = SnakeEnv()
    env.reset(seed=0)
    env.game.load_state(snake=[(4, 2), (4, 1), (4, 0)], direction=RIGHT, fruits=FAR_FRUITS, obstacles=[(4, 3)])
    # À frente há obstáculo; para trás é meia-volta (ignorada = seguir reto = morrer).
    assert env.action_masks().tolist() == [True, False, True, False]


def test_render_rgb_array():
    env = SnakeEnv(BOARDS["mini"], render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------
def test_q_update_by_hand():
    """Uma atualização calculada à mão: Q = 0 + 0,5 * (1 + 0,9 * 2 - 0) = 1,4."""
    agent = QLearningAgent("features", alpha=0.5, gamma=0.9)
    s, s2 = np.zeros(11, np.float32), np.ones(11, np.float32)
    agent.q[agent.key(s2)] = np.array([0.0, 2.0, 0.0, 0.0])
    error = agent.learn(s, RIGHT, reward=1.0, next_obs=s2, terminated=False)
    assert error == pytest.approx(2.8)
    assert agent.q[agent.key(s)][RIGHT] == pytest.approx(1.4)
    # Se a partida terminou, o futuro não conta: alvo = só a recompensa.
    agent2 = QLearningAgent("features", alpha=0.5, gamma=0.9)
    agent2.q[agent2.key(s2)] = np.array([0.0, 2.0, 0.0, 0.0])
    agent2.learn(s, RIGHT, reward=-1.0, next_obs=s2, terminated=True)
    assert agent2.q[agent2.key(s)][RIGHT] == pytest.approx(-0.5)


def test_choose_picks_best_and_counts_unseen():
    agent = QLearningAgent("features", seed=0)
    s = np.zeros(11, np.float32)
    agent.q[agent.key(s)] = np.array([0.1, 0.9, 0.2, 0.0])
    assert all(agent.choose(s, epsilon=0.0) == (RIGHT, True) for _ in range(20))
    action, seen = agent.choose(np.ones(11, np.float32), epsilon=0.0)
    assert not seen and action in (UP, RIGHT, DOWN, LEFT)


def test_qlearning_learns_on_mini_board():
    """Treino curto com o resumo de 11 números: tem que bater de longe o aleatório."""
    config = BOARDS["mini"]
    env = SnakeEnv(config, "features")
    agent = QLearningAgent("features", seed=0)
    rng = np.random.default_rng(0)
    episodes = 3000
    for ep in range(episodes):
        epsilon = max(0.05, 1 - ep / (0.8 * episodes))
        obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
        done = False
        while not done:
            action, _ = agent.choose(obs, epsilon)
            nxt, reward, term, trunc, _ = env.step(action)
            agent.learn(obs, action, reward, nxt, term)
            obs, done = nxt, term or trunc
    trained = np.mean([r["fruits"] for r in evaluate(agent, eval_seeds(100), config)])
    assert trained > 5  # o aleatório faz 0,34 no mini


def test_save_and_load_roundtrip(tmp_path):
    agent = QLearningAgent("grid", alpha=0.2, gamma=0.8)
    agent.q[b"abc"] = np.array([1.0, 2.0, 3.0, 4.0])
    agent.save(tmp_path / "q.pkl")
    loaded = QLearningAgent.load(tmp_path / "q.pkl")
    assert (loaded.obs_type, loaded.alpha, loaded.gamma) == ("grid", 0.2, 0.8)
    assert loaded.q[b"abc"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_agent_works_with_play_game():
    result = play_game(QLearningAgent("features", seed=0), 123, BOARDS["mini"])
    assert result["steps"] == len(result["actions"]) > 0
