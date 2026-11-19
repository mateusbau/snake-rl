"""Testes da Fase 6: máscara de ações e agente PPO (Stable-Baselines3), sempre na CPU."""

import numpy as np

from agents.ppo import PPOAgent, train_quick
from snake.env import RIGHT, UP, SnakeGame
from snake.evaluation import eval_seeds, evaluate
from snake.gym_env import SnakeEnv

FAR_FRUITS = [(8, 9), (8, 7), (8, 5), (6, 9), (8, 3)]


def trapped_env() -> SnakeEnv:
    """Cabeça H no canto (0,0), olhando para cima, sem saída:
        H o o        cima/esquerda = parede; direita (0,1) = corpo;
        o o .        baixo = meia-volta (vale "seguir reto" = parede).
    """
    env = SnakeEnv()
    env.reset(seed=0)
    env.game.load_state(snake=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)], direction=UP, fruits=FAR_FRUITS)
    return env


def test_mask_when_trapped_allows_everything():
    # Nenhuma jogada é segura. Uma máscara toda False quebraria o MaskablePPO
    # (probabilidades 0/0); como qualquer ação perde, liberamos todas.
    env = trapped_env()
    assert env.game.safe_directions() == []
    assert env.action_masks().tolist() == [True, True, True, True]


def test_mask_normal_case_unchanged():
    env = SnakeEnv()
    env.reset(seed=0)
    env.game.load_state(snake=[(4, 2), (4, 1), (4, 0)], direction=RIGHT, fruits=FAR_FRUITS, obstacles=[(4, 3)])
    assert env.action_masks().tolist() == [True, False, True, False]


def test_masked_agent_never_picks_an_unsafe_move():
    # Rede recém-criada (sem treino, decisões quase aleatórias): mesmo assim, com a
    # máscara ela só escolhe direções seguras, em todos os passos de várias partidas.
    agent = PPOAgent.new("features_space", masked=True, seed=0)
    for seed in eval_seeds(5):
        game = SnakeGame(seed=seed)
        agent.reset(seed)
        while not game.game_over:
            safe = game.safe_directions()
            action = agent.act(game)
            if safe:
                assert action in safe
            game.step(action)


def test_ppo_learns_on_mini_board():
    """Treino curtinho no 5×5 (sem máscara): tem que sair do nível do aleatório (0,34)."""
    agent = train_quick("mini", "features", steps=20_000, masked=False, seed=0)
    results = evaluate(agent, eval_seeds(50), agent.config)
    assert np.mean([r["fruits"] for r in results]) > 2


def test_save_and_load_roundtrip(tmp_path):
    agent = PPOAgent.new("features_space", masked=True, seed=1)
    agent.save(tmp_path / "model.zip")
    loaded = PPOAgent.load(tmp_path / "model.zip")
    assert loaded.obs_type == "features_space" and loaded.masked
    game = SnakeGame(seed=9)
    obs = loaded.observe(game)
    a = agent.model.policy.predict_values(agent.model.policy.obs_to_tensor(obs)[0])
    b = loaded.model.policy.predict_values(loaded.model.policy.obs_to_tensor(obs)[0])
    assert np.allclose(a.detach().numpy(), b.detach().numpy())


# ---------------------------------------------------------------------------
# Currículo: partidas de treino que começam já avançadas
# ---------------------------------------------------------------------------
from snake.curriculum import collect_start_states  # noqa: E402

START = {"snake": [(4, 5), (4, 4), (4, 3), (4, 2)], "direction": RIGHT,
         "fruits": FAR_FRUITS, "obstacles": [(2, 2)], "fruits_collected": 1}


def test_env_starts_from_a_given_state():
    env = SnakeEnv(start_states=[START], start_prob=1.0)
    _, info = env.reset(seed=3)
    assert list(env.game.snake) == START["snake"] and env.game.obstacles == [(2, 2)]
    assert info["fruits"] == 1 and env.game.steps == 0
    env.step(RIGHT)  # o jogo segue normalmente a partir dali
    assert env.game.head == (4, 6)


def test_env_without_curriculum_starts_normally():
    env = SnakeEnv(start_states=[START], start_prob=0.0)
    env.reset(seed=3)
    assert list(env.game.snake) == list(SnakeGame(seed=3).snake)


def test_collect_start_states_are_valid_and_avoid_eval_seeds():
    states, seeds = collect_start_states(n_games=3, seed=0, min_fruits=5, every=20)
    assert states and not set(seeds) & set(eval_seeds(1000))
    for s in states:
        assert s["fruits_collected"] >= 5
        game = SnakeGame(seed=0)
        game.load_state(**s)  # valida o estado (cobra contínua, sem sobreposição...)
        assert len(game.fruits) == 5
