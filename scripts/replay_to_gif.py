"""Transforma um replay (.json) num GIF animado, sem abrir janela.

Como rodar:
    .venv\\Scripts\\python.exe scripts\\replay_to_gif.py replays\\human\\ARQUIVO.json docs\\media\\minha_partida.gif
    .venv\\Scripts\\python.exe scripts\\replay_to_gif.py --demo docs\\media\\fase2_demo.gif   # partida de exemplo

Opções: --speed (passos por segundo no GIF), --scale (tamanho da imagem).
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["SDL_VIDEODRIVER"] = "dummy"  # desenha na memória, sem janela
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np  # noqa: E402
import pygame  # noqa: E402
from PIL import Image  # noqa: E402

from snake import replay as rp  # noqa: E402
from snake.env import SnakeGame  # noqa: E402
from snake.render import Renderer  # noqa: E402


def demo_replay(seed: int = 3, min_fruits: int = 12) -> rp.Replay:
    """Partida de exemplo de um jogador aleatório "que evita morte imediata".

    Tenta várias seeds até achar uma partida com pelo menos `min_fruits` frutas,
    para o GIF mostrar vários obstáculos surgindo.
    """
    while True:
        rng = np.random.default_rng(seed)
        game = SnakeGame(seed=seed)
        replay = rp.Replay(seed=seed, player="safe_random")
        while not game.game_over:
            action = int(rng.choice(game.safe_directions() or [game.direction]))
            replay.actions.append(action)
            game.step(action)
        if game.fruits_collected >= min_fruits:
            replay.result = rp.result_of(game)
            return replay
        seed += 1


def to_gif(replay: rp.Replay, out: Path, speed: float, scale: float) -> None:
    pygame.init()
    frames = []
    renderer = None
    surface = None
    for game in rp.frames(replay):
        if renderer is None:
            renderer = Renderer(game)
            surface = pygame.Surface((renderer.width, renderer.height))
        renderer.draw(surface, game, hud_right=replay.player)
        # pygame -> texto de bytes RGB -> imagem do Pillow
        image = Image.frombytes("RGB", surface.get_size(), pygame.image.tobytes(surface, "RGB"))
        if scale != 1:
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
        frames.append(image)
    pygame.quit()

    duration = int(1000 / speed)  # milissegundos por quadro
    durations = [duration] * len(frames)
    durations[-1] = 2000          # segura o último quadro por 2 s
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"GIF com {len(frames)} quadros salvo em {out} ({out.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path, nargs="?", help="arquivo .json do replay")
    parser.add_argument("out", type=Path, nargs="?", default=ROOT / "docs" / "media" / "replay.gif")
    parser.add_argument("--demo", type=Path, help="gera um GIF de exemplo neste caminho")
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0.6)
    args = parser.parse_args()
    if args.demo:
        to_gif(demo_replay(), args.demo, args.speed, args.scale)
    elif args.replay:
        to_gif(rp.load(args.replay), args.out, args.speed, args.scale)
    else:
        parser.error("informe um replay ou --demo")


if __name__ == "__main__":
    main()
