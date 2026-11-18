"""Testes da Fase 2: replays e desenho (sem abrir janela)."""

import os

import numpy as np

# "dummy" faz o pygame desenhar só na memória, sem janela (funciona em testes).
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame  # noqa: E402

from snake import replay as rp  # noqa: E402
from snake.env import RIGHT, SnakeGame  # noqa: E402
from snake.render import Renderer  # noqa: E402


def random_replay(seed: int) -> tuple[rp.Replay, str]:
    """Joga uma partida com ações aleatórias (incluindo None = manter) e grava."""
    rng = np.random.default_rng(seed)
    game = SnakeGame(seed=seed)
    replay = rp.Replay(seed=seed, player="test")
    while not game.game_over:
        options = game.safe_directions() or [game.direction]
        action = None if rng.random() < 0.3 else int(rng.choice(options))
        replay.actions.append(action)
        game.step(action)
    replay.result = rp.result_of(game)
    return replay, str(game)


def test_replay_reproduces_game_exactly(tmp_path):
    # tmp_path é uma pasta temporária que o pytest cria e apaga sozinho.
    for seed in range(30):
        replay, final_board = random_replay(seed)
        path = tmp_path / f"r{seed}.json"
        rp.save(replay, path)
        loaded = rp.load(path)
        game = rp.simulate(loaded)  # confere o resultado gravado
        assert str(game) == final_board
        assert loaded.actions == replay.actions


def test_replay_file_is_compact(tmp_path):
    replay, _ = random_replay(0)
    path = tmp_path / "r.json"
    rp.save(replay, path)
    text = path.read_text(encoding="utf-8")
    assert '"actions": [' in text and text.count("\n") < 60  # ações numa linha só


def test_render_draws_expected_colors():
    pygame.init()
    game = SnakeGame(seed=0)
    for _ in range(3):
        game.step(RIGHT)  # come uma fruta e nasce um obstáculo
    renderer = Renderer(game)
    surface = pygame.Surface((renderer.width, renderer.height))
    renderer.draw(surface, game)
    cfg = renderer.cfg

    def color_at(cell, dx=0, dy=0):
        rect = renderer.cell_rect(cell)
        return tuple(surface.get_at((rect.centerx + dx, rect.centery + dy)))[:3]

    assert color_at(game.obstacles[0]) == cfg.obstacle
    assert color_at(game.fruits[0]) == cfg.fruit
    assert color_at((8, 0)) in (cfg.light, cfg.dark) or (8, 0) in game.fruits + game.obstacles
    # Corpo azul (b é o maior canal de cor)
    r, g, b = color_at(game.snake[-1])
    assert b > r and b > g
    pygame.quit()
