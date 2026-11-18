"""Testes da Fase 5: observação de 18 números, replay buffer e agente DQN (sempre na CPU)."""

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from agents.dqn import CNN, MLP, DQNAgent, ReplayBuffer
from snake.boards import BOARDS
from snake.env import RIGHT, SnakeGame
from snake.evaluation import eval_seeds, evaluate
from snake.gym_env import SPACE_FEATURE_NAMES, SnakeEnv, feature_observation, space_feature_observation

FAR_FRUITS = [(8, 9), (8, 7), (0, 9), (8, 5), (0, 7)]


def test_env_with_space_features_passes_checker():
    check_env(SnakeEnv(obs_type="features_space").unwrapped, skip_render_check=True)


def test_space_features_start_with_the_old_11():
    game = SnakeGame(seed=3)
    obs = space_feature_observation(game)
    assert obs.shape == (18,) and obs.dtype == np.float32
    assert np.array_equal(obs[:11], feature_observation(game))


def test_space_features_see_a_pocket():
    # Cobra subindo pela coluna 1 (cabeça H em (1,1)), corpo dando a volta à direita:
    #   linha 0:  . . o o .
    #   linha 1:  . H o . .
    #   linha 2:  . o o . .
    #   linha 3:  . o o . .
    game = SnakeGame(seed=0)
    game.load_state(snake=[(1, 1), (2, 1), (3, 1), (3, 2), (2, 2), (1, 2), (0, 2), (0, 3)],
                    direction=0, fruits=FAR_FRUITS, obstacles=[])
    # À direita (1,2) é corpo: perigo, então espaço 0 e cauda inalcançável.
    f = dict(zip(SPACE_FEATURE_NAMES, space_feature_observation(game)))
    assert f["danger_right"] == 1 and f["space_right"] == 0 and f["tail_right"] == 0
    # À esquerda, (1,0): sai para o resto do tabuleiro -> quase todo o espaço livre.
    assert f["space_left"] > 0.9 and f["tail_left"] == 1
    # Reto, (0,1): também escapa por (0,0) e pela coluna 0.
    assert f["space_straight"] > 0.9
    assert f["length"] == pytest.approx(8 / 90)


def test_space_features_detect_dead_end():
    # Cabeça em (0,1) olhando para a esquerda; ir para (0,0) é um beco de 1 casa
    # (abaixo de (0,0) há obstáculo). Reto = beco; o espaço deve ser 0 lá.
    game = SnakeGame(seed=0)
    game.load_state(snake=[(0, 1), (0, 2), (0, 3)], direction=3, fruits=FAR_FRUITS, obstacles=[(1, 0), (1, 1)])
    f = dict(zip(SPACE_FEATURE_NAMES, space_feature_observation(game)))
    assert f["danger_straight"] == 0 and f["space_straight"] == 0 and f["tail_straight"] == 0
    # Direita (relativa a "esquerda") = cima = parede; esquerda relativa = baixo = obstáculo.
    assert f["danger_right"] == 1 and f["danger_left"] == 1


def test_replay_buffer_is_circular():
    buf = ReplayBuffer(capacity=3, obs_shape=(2,), obs_dtype=np.float32, seed=0)
    for i in range(5):  # 5 jogadas numa memória de 3: sobram as 3 últimas (2, 3, 4)
        buf.add(np.full(2, i), action=i % 4, reward=float(i), next_obs=np.full(2, i + 1), terminated=False)
    assert len(buf) == 3
    assert sorted(buf.rewards.tolist()) == [2.0, 3.0, 4.0]
    batch = buf.sample(10)
    assert batch["obs"].shape == (10, 2) and set(batch["rewards"]) <= {2.0, 3.0, 4.0}


def test_networks_output_four_values():
    assert MLP(18)(torch.zeros(5, 18)).shape == (5, 4)
    assert CNN(9, 10)(torch.zeros(5, 4, 9, 10, dtype=torch.uint8)).shape == (5, 4)


def test_train_step_moves_q_toward_target():
    # Uma única jogada repetida: estado s, ação 2, recompensa 1, fim. O alvo é 1,
    # então depois de vários passos de gradiente Q(s, 2) tem que chegar perto de 1.
    agent = DQNAgent("features", 9, 10, lr=1e-2, seed=0)
    obs = np.zeros(11, dtype=np.float32)
    obs[[0, 4, 8]] = 1
    batch = {"obs": np.stack([obs] * 32), "actions": np.full(32, 2), "rewards": np.ones(32, dtype=np.float32),
             "next_obs": np.stack([obs] * 32), "terminated": np.ones(32, dtype=np.float32)}
    for _ in range(200):
        agent.train_step(batch)
    assert agent.q_values(obs)[2] == pytest.approx(1.0, abs=0.05)


def test_dqn_learns_on_mini_board():
    """Treino curtinho no 5×5: tem que sair do nível do aleatório (0,34 frutas)."""
    config = BOARDS["mini"]
    env = SnakeEnv(config, obs_type="features")
    agent = DQNAgent("features", 5, 5, lr=1e-3, seed=0)
    buffer = ReplayBuffer(20_000, (11,), np.float32, seed=0)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    for step in range(1, 6001):
        epsilon = max(0.05, 1 - step / 3000)
        action = agent.choose(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        if step >= 500 and step % 2 == 0:
            agent.train_step(buffer.sample(64))
        if step % 500 == 0:
            agent.sync_target()
    results = evaluate(agent, eval_seeds(50), config)
    assert np.mean([r["fruits"] for r in results]) > 3


def test_save_and_load_roundtrip(tmp_path):
    agent = DQNAgent("features_space", 9, 10, seed=1)
    path = tmp_path / "m.pt"
    agent.save(path)
    loaded = DQNAgent.load(path)
    obs = space_feature_observation(SnakeGame(seed=5))
    assert np.allclose(agent.q_values(obs), loaded.q_values(obs))
    assert loaded.obs_type == "features_space"


def test_dqn_agent_plays_a_full_game():
    agent = DQNAgent("grid", 9, 10, seed=0)
    game = SnakeGame(seed=0)
    game.load_state(snake=[(4, 2), (4, 1), (4, 0)], direction=RIGHT, fruits=FAR_FRUITS, obstacles=[])
    assert agent.act(game) in (0, 1, 2, 3)
    (result,) = evaluate(agent, eval_seeds(1))
    assert result["steps"] > 0 and result["end_reason"] is not None
