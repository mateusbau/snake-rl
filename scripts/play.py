"""Jogue Snake com obstáculos no teclado (Fase 2). Toda partida terminada é gravada.

Como rodar (na raiz do projeto):
    .venv\\Scripts\\python.exe scripts\\play.py                 # jogar
    .venv\\Scripts\\python.exe scripts\\play.py --speed 6       # mais devagar
    .venv\\Scripts\\python.exe scripts\\play.py --replay replays\\human\\ARQUIVO.json   # assistir

Controles:
    Setas ou WASD   mover (a cobra só começa a andar no primeiro movimento)
    Espaço / P      pausar
    Enter           nova partida (depois de perder)
    Esc             sair

Os replays vão para replays/human/. Veja suas estatísticas com:
    .venv\\Scripts\\python.exe scripts\\human_stats.py
"""

import argparse
import os
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from snake import replay as rp  # noqa: E402
from snake.env import DOWN, LEFT, OPPOSITE, RIGHT, UP, SnakeGame  # noqa: E402
from snake.render import Renderer  # noqa: E402

REPLAY_DIR = ROOT / "replays" / "human"

# Teclas -> direção. Um dicionário funciona como um objeto { tecla: direção } do JS.
KEYS = {
    pygame.K_UP: UP, pygame.K_w: UP,
    pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
    pygame.K_DOWN: DOWN, pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT, pygame.K_a: LEFT,
}
END_TEXT = {
    "wall": "Bateu na parede",
    "body": "Bateu no próprio corpo",
    "obstacle": "Bateu num obstáculo",
    "timeout": "Ficou 200 passos sem comer",
    "win": "ZEROU O JOGO!",
}


def human_record() -> int:
    """Maior número de frutas entre os replays humanos já gravados."""
    best = 0
    for path in REPLAY_DIR.glob("*.json"):
        best = max(best, rp.load(path).result.get("fruits", 0))
    return best


def play(speed: float, seed: int | None) -> None:
    pygame.init()
    game = SnakeGame()  # só para o Renderer saber o tamanho do tabuleiro
    renderer = Renderer(game)
    screen = pygame.display.set_mode((renderer.width, renderer.height))
    pygame.display.set_caption("Snake RL — você joga")
    clock = pygame.time.Clock()
    interval = 1.0 / speed            # segundos entre um passo e outro
    record = human_record()
    games_this_session = 0

    def new_game():
        # Cada partida tem sua seed, sorteada e gravada no replay.
        game_seed = seed if seed is not None else random.randrange(2**31)
        return SnakeGame(seed=game_seed), rp.Replay(seed=game_seed, player="human"), game_seed

    game, replay, game_seed = new_game()
    state = "waiting"                  # waiting -> playing <-> paused -> over
    # Fila de comandos: guarda até 3 teclas apertadas entre dois passos, para
    # curvas rápidas (ex.: ↑ e ← quase juntas) não se perderem.
    queue: deque[int] = deque()
    last_step = time.perf_counter()

    running = True
    while running:
        # ---- 1) Eventos (teclado, fechar janela) ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in KEYS and state in ("waiting", "playing"):
                    direction = KEYS[event.key]
                    last = queue[-1] if queue else game.direction
                    # Ignora repetição e meia-volta já na fila (o motor também ignoraria).
                    if direction != last and direction != OPPOSITE[last] and len(queue) < 3:
                        queue.append(direction)
                    if state == "waiting":  # qualquer seta começa a partida
                        state = "playing"
                        last_step = time.perf_counter() - interval  # anda já no 1º toque
                elif event.key in (pygame.K_SPACE, pygame.K_p) and state in ("playing", "paused"):
                    state = "paused" if state == "playing" else "playing"
                    last_step = time.perf_counter()
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and state == "over":
                    game, replay, game_seed = new_game()
                    queue.clear()
                    state = "waiting"

        # ---- 2) Passos do jogo, no ritmo de `speed` passos por segundo ----
        now = time.perf_counter()
        while state == "playing" and now - last_step >= interval:
            last_step += interval
            action = queue.popleft() if queue else None
            replay.actions.append(action)
            result = game.step(action)
            if result.game_over:
                state = "over"
                replay.result = rp.result_of(game)
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                path = REPLAY_DIR / f"{stamp}_seed{game_seed}.json"
                rp.save(replay, path)
                games_this_session += 1
                new_record = game.fruits_collected > record
                record = max(record, game.fruits_collected)
                print(f"Partida {games_this_session}: {game.fruits_collected} frutas "
                      f"({game.end_reason}) -> replay {path.name}")

        # ---- 3) Desenho ----
        renderer.draw(screen, game, hud_right=f"recorde {record}")
        if state == "waiting":
            renderer.draw_message(screen, "Snake RL", ["Aperte uma seta para começar", "Espaço pausa · Esc sai"])
        elif state == "paused":
            renderer.draw_message(screen, "Pausado", ["Espaço para continuar"])
        elif state == "over":
            title = "Novo recorde!" if new_record else END_TEXT.get(game.end_reason, "Fim")
            renderer.draw_message(screen, title, [
                f"{game.fruits_collected} frutas · {game.obstacles_generated} obstáculos",
                f"Partidas nesta sessão: {games_this_session}",
                "Enter: jogar de novo · Esc: sair",
            ])
        pygame.display.flip()
        clock.tick(120)  # redesenha até 120x/s (o jogo em si anda no ritmo de `speed`)

    pygame.quit()


def watch(path: Path, speed: float) -> None:
    """Assiste a um replay gravado."""
    pygame.init()
    replay = rp.load(path)
    frames = rp.frames(replay)
    game = next(frames)
    renderer = Renderer(game)
    screen = pygame.display.set_mode((renderer.width, renderer.height))
    pygame.display.set_caption(f"Replay — {path.name}")
    clock = pygame.time.Clock()
    finished = False
    while True:
        if any(e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
               for e in pygame.event.get()):
            break
        if not finished:
            # next(frames, None) pega o próximo quadro, ou None quando acabar.
            finished = next(frames, None) is None
        renderer.draw(screen, game, hud_right=f"replay · {replay.player}")
        if finished:
            renderer.draw_message(screen, END_TEXT.get(game.end_reason, "Fim"),
                                  [f"{game.fruits_collected} frutas", "Esc para sair"])
        pygame.display.flip()
        clock.tick(speed)
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed", type=float, default=7.0, help="passos por segundo (padrão 7)")
    parser.add_argument("--seed", type=int, default=None, help="fixa a seed de todas as partidas")
    parser.add_argument("--replay", type=Path, default=None, help="assistir a um replay em vez de jogar")
    args = parser.parse_args()
    if args.replay:
        watch(args.replay, args.speed)
    else:
        play(args.speed, args.seed)


if __name__ == "__main__":
    main()
