"""Gera o vídeo do LinkedIn (Fase 8): MP4 quadrado 1080×1080, ~60 s, com legendas.

Roteiro (o do CLAUDE.md):
  1. gancho: a rede com a cobra enorme, quase zerando
  2. obstáculos surgindo a cada fruta ímpar
  3. tela dividida em 4: a MESMA partida em 4 momentos do treino
  4. curva de aprendizado animada (com a minha média humana de referência)
  5. barras: aleatório, guloso, eu, DQN, PPO, BFS, PPO + busca
  6. a primeira vitória (1 em 1.000) e o limite da paridade
  7. cartão final com os links

Como funciona: cada quadro é desenhado com pygame numa Surface (o "canvas"), vira
bytes RGB e é empurrado para o ffmpeg por um "pipe" (um cano entre dois programas:
o Python escreve de um lado, o ffmpeg lê do outro e comprime em H.264). O ffmpeg
vem dentro do pacote imageio-ffmpeg, então não precisa instalar nada à parte.

Tudo sai dos arquivos do projeto: replays (seed + ações), checkpoints, metrics.csv e
summary.json. Nada é digitado à mão, então o vídeo sempre bate com os relatórios.

Como rodar (~1 min):
    .venv\\Scripts\\python.exe scripts\\make_video.py
    .venv\\Scripts\\python.exe scripts\\make_video.py --preview   # só salva 1 quadro PNG por cena
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["SDL_VIDEODRIVER"] = "dummy"          # desenha na memória, sem janela
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from agents.ppo import PPOAgent  # noqa: E402
from snake import replay as rp  # noqa: E402
from snake.env import SnakeGame  # noqa: E402
from snake.render import Renderer, RenderConfig  # noqa: E402

SIZE = 1080
FPS = 30
CAPTION_H = 200                       # faixa de legenda no topo
BG = (27, 42, 18)                     # verde quase preto (fundo)
INK = (255, 255, 255)
MUTED = (190, 205, 175)
ACCENT = (123, 156, 255)              # azul da cobra, mais claro (destaques)
HUMAN = (255, 179, 71)                # laranja: tudo que é "eu, humano"

GITHUB = "github.com/mateusbau/snake-rl"
DEMO = "mateusbau.github.io/snake-rl"

CHECKPOINTS = [                        # (rótulo, arquivo) para a tela dividida
    ("sem treino", "experiments/fase6_ppo/checkpoints/ppo_step0.zip"),
    ("100 mil passos", "experiments/fase6_ppo/checkpoints/ppo_step100000.zip"),
    ("1 milhão de passos", "experiments/fase6_ppo/checkpoints/ppo_step1000000.zip"),
    ("4 milhões de passos", "experiments/fase6_ppo/model.zip"),
]
SPLIT_SEED = 2056                      # escolhida por ter um fim de 4 milhões bonito (58 frutas)


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------
def font(size: int, bold: bool = False) -> pygame.font.Font:
    """Segoe UI (Windows) se existir; senão, a fonte padrão do pygame."""
    path = Path("C:/Windows/Fonts") / ("segoeuib.ttf" if bold else "segoeui.ttf")
    return pygame.font.Font(str(path) if path.exists() else None, size)


def text(surface, msg: str, size: int, pos, color=INK, bold=False, anchor="center") -> pygame.Rect:
    image = font(size, bold).render(msg, True, color)
    rect = image.get_rect(**{anchor: pos})
    surface.blit(image, rect)
    return rect


def caption(surface, title: str, subtitle: str = "") -> None:
    """Legenda no topo: título grande e uma linha menor de explicação."""
    pygame.draw.rect(surface, BG, (0, 0, SIZE, CAPTION_H))
    text(surface, title, 58, (SIZE // 2, 78 if subtitle else CAPTION_H // 2), bold=True)
    if subtitle:
        text(surface, subtitle, 34, (SIZE // 2, 146), MUTED)


# ---------------------------------------------------------------------------
# Saída: ffmpeg por pipe
# ---------------------------------------------------------------------------
class Video:
    """Recebe Surfaces e manda para o ffmpeg. `fade` faz a transição entre cenas."""

    FADE = 10   # quadros de transição (1/3 de segundo)

    def __init__(self, out: Path, preview: bool):
        self.preview = preview
        self.out = out
        self.frames = 0
        self.last: pygame.Surface | None = None
        self.fading = 0
        self.scene = 0
        self.proc = None
        if not preview:
            import imageio_ffmpeg

            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-",
                   # yuv420p + faststart: o formato que LinkedIn, WhatsApp e navegadores aceitam sem reclamar.
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
                   "-movflags", "+faststart", str(out)]
            # stdin=PIPE: o que escrevermos em proc.stdin chega no "-i -" do ffmpeg.
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def save_preview(self) -> None:
        """Modo prévia: salva o último quadro da cena que acabou de terminar."""
        if self.preview and self.last is not None:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(self.last, str(self.out.parent / f"video_cena{self.scene}_{self.scene_name}.png"))

    def new_scene(self, name: str) -> None:
        self.save_preview()
        self.scene += 1
        self.scene_name = name
        if self.last is not None:
            self.fading = self.FADE

    def add(self, surface: pygame.Surface, repeat: int = 1) -> None:
        if self.preview:
            # Modo prévia: não grava vídeo; save_preview() salva o último quadro de cada cena.
            self.frames += repeat
            self.last = surface.copy()
            return
        for _ in range(repeat):
            frame = surface
            if self.fading:
                # Mistura o último quadro da cena anterior com o novo (dissolve).
                frame = surface.copy()
                old = self.last.copy()
                old.set_alpha(int(255 * self.fading / (self.FADE + 1)))
                frame.blit(old, (0, 0))
                self.fading -= 1
            self.proc.stdin.write(pygame.image.tobytes(frame, "RGB"))
            self.frames += 1
        self.last = surface.copy()

    def close(self) -> None:
        self.save_preview()
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        print(f"{self.frames} quadros = {self.frames / FPS:.1f} s" + ("" if self.preview else f" -> {self.out}"))


# ---------------------------------------------------------------------------
# Tabuleiros
# ---------------------------------------------------------------------------
BIG = RenderConfig(cell_size=80)
SMALL = RenderConfig(cell_size=39, border=10, hud_height=56)


def board_surface(game: SnakeGame, cfg: RenderConfig, label: str = "", flash=None, overlay=None) -> pygame.Surface:
    """Desenha um jogo numa Surface própria. `flash` = célula a destacar (obstáculo novo)."""
    renderer = Renderer(game, cfg)
    surface = pygame.Surface((renderer.width, renderer.height))
    renderer.draw(surface, game, hud_right=label)
    if flash:
        cell, strength = flash
        rect = renderer.cell_rect(cell).inflate(10, 10)
        pygame.draw.rect(surface, (255, 255, 255), rect, width=max(2, int(8 * strength)), border_radius=6)
    if overlay:
        # Etiqueta no meio do tabuleiro (em vez de escurecer tudo: o tabuleiro continua visível).
        size = max(24, cfg.cell_size * 2 // 3)
        image = font(size, bold=True).render(overlay, True, INK)
        box = image.get_rect(center=(renderer.width // 2, renderer.height // 2 + cfg.hud_height // 2)).inflate(size, size // 2)
        pill = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(pill, (20, 30, 14, 215), pill.get_rect(), border_radius=box.height // 2)
        surface.blit(pill, box)
        surface.blit(image, image.get_rect(center=box.center))
    return surface


def frame_with_board(board: pygame.Surface, title: str, subtitle: str = "", badge: str = "") -> pygame.Surface:
    surface = pygame.Surface((SIZE, SIZE))
    surface.fill(BG)
    caption(surface, title, subtitle)
    x = (SIZE - board.get_width()) // 2
    y = CAPTION_H + (SIZE - CAPTION_H - board.get_height()) // 2
    surface.blit(board, (x, y))
    if badge:
        text(surface, badge, 30, (SIZE - 40, SIZE - 30), MUTED, anchor="midright")
    return surface


def replay_states(replay: rp.Replay):
    """Lista de (jogo copiado, resultado do passo) para poder andar para frente e para trás."""
    game = SnakeGame(replay.game_config(), seed=replay.seed)
    states = [(snapshot(game), None)]
    for action in replay.actions:
        result = game.step(action)
        states.append((snapshot(game), result))
    return states


def snapshot(game: SnakeGame) -> SnakeGame:
    """Cópia leve do jogo (só o que o desenho usa)."""
    copy = SnakeGame(game.config, seed=0)
    copy.load_state(game.snake, game.direction, game.fruits, game.obstacles, game.fruits_collected)
    copy.obstacles_generated = game.obstacles_generated
    copy.steps, copy.game_over, copy.end_reason = game.steps, game.game_over, game.end_reason
    return copy


# ---------------------------------------------------------------------------
# Cenas
# ---------------------------------------------------------------------------
def scene_hook(video: Video) -> None:
    """1. A rede com a cobra enorme. Começa no meio da partida, quando já está lotado."""
    replay = rp.load(ROOT / "experiments" / "fase6_comparacao" / "replays" / "ppo_best.json")
    states = replay_states(replay)
    last_meal = max(i for i, (_, r) in enumerate(states) if r is not None and r.ate_fruit)
    start = next(i for i, (g, _) in enumerate(states) if g.fruits_collected >= 46)
    video.new_scene("gancho")
    for i in range(start, last_meal + 1, 2):         # 2 passos por quadro = 60 passos/s
        g = states[i][0]
        video.add(frame_with_board(board_surface(g, BIG), "Uma IA aprendeu Snake sozinha",
                                   "ninguém ensinou as regras: só recompensa por fruta", "acelerado 6×"))
    g = states[last_meal][0]
    video.add(frame_with_board(board_surface(g, BIG), "Uma IA aprendeu Snake sozinha",
                               f"{g.fruits_collected} frutas, cobra de {len(g.snake)} casas"), repeat=45)


def scene_obstacles(video: Video) -> None:
    """2. O começo de uma partida, piscando cada obstáculo que nasce."""
    agent = PPOAgent.load(ROOT / "experiments" / "fase6_ppo" / "model.zip")
    game = SnakeGame(seed=SPLIT_SEED)
    video.new_scene("obstaculos")
    flash, strength = None, 0.0
    while game.fruits_collected < 13 and not game.game_over:
        result = game.step(agent.act(game))
        if result.new_obstacle:
            flash, strength = result.new_obstacle, 1.0
        for _ in range(3):                            # 1 passo a cada 3 quadros = 10 passos/s
            board = board_surface(game, BIG, flash=(flash, strength) if strength > 0 else None)
            video.add(frame_with_board(board, "A cada fruta ímpar, um obstáculo",
                                       "permanente, em lugar sorteado: o mapa vai fechando"))
            strength = max(0.0, strength - 0.05)
    video.add(video.last, repeat=20)


def scene_split(video: Video) -> None:
    """3. A mesma seed jogada por 4 checkpoints, lado a lado."""
    runs = []
    for label, path in CHECKPOINTS:
        agent = PPOAgent.load(ROOT / path)
        game = SnakeGame(seed=SPLIT_SEED)
        states = [snapshot(game)]
        while not game.game_over:
            game.step(agent.act(game))
            states.append(snapshot(game))
        runs.append((label, states))
    longest = max(len(s) for _, s in runs)

    # Velocidade variável: devagar no começo (as 3 primeiras morrem rápido), depois acelera.
    timeline, t = [], 0.0
    while t < longest - 1:
        timeline.append(int(t))
        t += 0.5 if t < 60 else 3.0
    timeline += [longest - 1] * 75                    # segura o final 2,5 s

    end_text = {"wall": "parede", "body": "corpo", "obstacle": "obstáculo", "timeout": "tempo", "win": "ZEROU"}
    video.new_scene("divisao")
    for step in timeline:
        surface = pygame.Surface((SIZE, SIZE))
        surface.fill(BG)
        caption(surface, "Mesma partida, 4 momentos do treino", "a mesma seed jogada por versões cada vez mais treinadas")
        for k, (label, states) in enumerate(runs):
            g = states[min(step, len(states) - 1)]
            over = f"{g.fruits_collected} frutas ({end_text[g.end_reason]})" if g.game_over else None
            board = board_surface(g, SMALL, label=label, overlay=over)
            cx = (k % 2) * SIZE // 2 + SIZE // 4
            cy = CAPTION_H + 222 + (k // 2) * 435
            surface.blit(board, board.get_rect(center=(cx, cy)))
        video.add(surface)


def scene_curve(video: Video) -> None:
    """4. Curva de aprendizado desenhada aos poucos, com a minha média como referência."""
    rows = list(csv.DictReader((ROOT / "experiments" / "fase6_ppo" / "metrics.csv").open(encoding="utf-8")))
    xs = [int(r["step"]) for r in rows]
    ys = [float(r["fruits_mean"]) for r in rows]
    human = json.loads((ROOT / "docs" / "human_baseline.json").read_text(encoding="utf-8"))["fruits_mean"]
    left, right, top, bottom = 150, SIZE - 70, CAPTION_H + 70, SIZE - 130
    x_max, y_max = 4_000_000, 60

    def px(x, y):
        return left + (right - left) * x / x_max, bottom - (bottom - top) * y / y_max

    frames = 6 * FPS
    video.new_scene("curva")
    for f in range(frames + 60):
        n = max(2, int(len(xs) * min(1.0, f / frames)))
        surface = pygame.Surface((SIZE, SIZE))
        surface.fill(BG)
        caption(surface, "Aprendendo com 4 milhões de passos", "frutas por partida durante o treino (PPO, 8 partidas em paralelo)")
        for y in range(0, y_max + 1, 10):              # grade horizontal discreta
            a, b = px(0, y), px(x_max, y)
            pygame.draw.line(surface, (52, 70, 40), a, b, 1)
            text(surface, str(y), 28, (left - 18, a[1]), MUTED, anchor="midright")
        for m in range(0, 5):
            a = px(m * 1_000_000, 0)
            text(surface, f"{m} mi" if m else "0", 28, (a[0], bottom + 30), MUTED)
        text(surface, "passos de treino", 28, ((left + right) // 2, bottom + 72), MUTED)
        # Linha da minha média (tracejada, laranja).
        hy = px(0, human)[1]
        for x0 in range(left, right, 24):
            pygame.draw.line(surface, HUMAN, (x0, hy), (min(x0 + 12, right), hy), 3)
        text(surface, f"minha média: {human:.1f}".replace(".", ","), 30, (right, hy + 24), HUMAN, anchor="midright")
        points = [px(x, y) for x, y in zip(xs[:n], ys[:n])]
        pygame.draw.lines(surface, ACCENT, False, points, 4)
        pygame.draw.circle(surface, ACCENT, points[-1], 8)
        text(surface, f"{ys[n - 1]:.1f}".replace(".", ","), 34, (points[-1][0] - 14, points[-1][1] - 34),
             INK, bold=True, anchor="midright")
        video.add(surface)


def scene_bars(video: Video) -> None:
    """5. Comparação final nas 1.000 sementes oficiais (barras crescendo)."""
    fase6 = json.loads((ROOT / "experiments" / "fase6_comparacao" / "summary.json").read_text(encoding="utf-8"))["agents"]
    fase7 = json.loads((ROOT / "experiments" / "fase7_comparacao" / "summary.json").read_text(encoding="utf-8"))["agents"]
    human = json.loads((ROOT / "docs" / "human_baseline.json").read_text(encoding="utf-8"))["fruits_mean"]
    bars = [
        ("Aleatório", fase6["random"]["fruits_mean"], MUTED),
        ("Eu (humano)", human, HUMAN),
        ("Guloso (regra)", fase6["greedy"]["fruits_mean"], MUTED),
        ("DQN (rede)", fase6["dqn_18"]["fruits_mean"], ACCENT),
        ("PPO (rede)", fase6["ppo"]["fruits_mean"], ACCENT),
        ("BFS (busca)", fase6["bfs"]["fruits_mean"], MUTED),
        ("PPO + busca", fase7["ppo_caca_fim"]["fruits_mean"], ACCENT),
    ]
    top, row = CAPTION_H + 80, 108
    left, right = 330, SIZE - 150
    grow = 2 * FPS
    video.new_scene("barras")
    for f in range(grow + 5 * FPS):
        t = min(1.0, f / grow)
        t = 1 - (1 - t) ** 3                           # "ease-out": desacelera no fim
        surface = pygame.Surface((SIZE, SIZE))
        surface.fill(BG)
        caption(surface, "Média de frutas em 1.000 partidas", "as mesmas 1.000 seeds para todos (as minhas: 20 partidas)")
        for i, (name, value, color) in enumerate(bars):
            y = top + i * row
            text(surface, name, 36, (left - 24, y + 30), INK if color != MUTED else MUTED, anchor="midright")
            width = max(4, (right - left) * value / 60 * t)
            pygame.draw.rect(surface, color, (left, y + 8, width, 44), border_radius=6)
            text(surface, f"{value * t:.1f}".replace(".", ","), 36, (left + width + 14, y + 30), INK, bold=True, anchor="midleft")
        if t >= 1:
            text(surface, "azul = aprendeu sozinha    cinza = regras escritas à mão", 28,
                 (SIZE // 2, SIZE - 50), MUTED)
        video.add(surface)


def scene_win(video: Video) -> None:
    """6. A primeira vitória e por que ela é rara."""
    replay = rp.load(ROOT / "experiments" / "fase7_eval_outros" / "replays" / "bfs_fim_best.json")
    states = replay_states(replay)
    start = next(i for i, (g, _) in enumerate(states) if g.fruits_collected >= 60)
    video.new_scene("vitoria")
    for i in range(start, len(states)):
        g = states[i][0]
        video.add(frame_with_board(board_surface(g, BIG), "Zerar: 1 vitória em 1.000",
                                   "busca + solucionador exato de fim de jogo"), repeat=2)
    g = states[-1][0]
    final = frame_with_board(board_surface(g, BIG, overlay="ZEROU!"), "Zerar: 1 vitória em 1.000",
                             "em ~64% das partidas é matematicamente impossível")
    video.add(final, repeat=4 * FPS)


def scene_end(video: Video) -> None:
    """7. Cartão final."""
    surface = pygame.Surface((SIZE, SIZE))
    surface.fill(BG)
    text(surface, "Snake RL", 110, (SIZE // 2, 300), bold=True)
    text(surface, "Python · PyTorch · Stable-Baselines3 · ONNX", 36, (SIZE // 2, 400), MUTED)
    text(surface, "Jogue contra a rede no navegador:", 38, (SIZE // 2, 560), INK)
    text(surface, DEMO, 46, (SIZE // 2, 625), ACCENT, bold=True)
    text(surface, "Código e relatórios de cada fase:", 38, (SIZE // 2, 740), INK)
    text(surface, GITHUB, 46, (SIZE // 2, 805), ACCENT, bold=True)
    video.new_scene("final")
    video.add(surface, repeat=4 * FPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "media" / "snake_rl.mp4")
    parser.add_argument("--preview", action="store_true", help="salva 1 PNG por cena em vez do vídeo")
    args = parser.parse_args()
    pygame.init()
    video = Video(args.out, args.preview)
    for scene in (scene_hook, scene_obstacles, scene_split, scene_curve, scene_bars, scene_win, scene_end):
        scene(video)
        print(f"cena {video.scene} ({video.scene_name}): até {video.frames / FPS:.1f} s")
    video.close()
    pygame.quit()


if __name__ == "__main__":
    main()
