"""Desenho do jogo com pygame (Fase 2).

Este arquivo só DESENHA: recebe um SnakeGame e pinta numa "Surface" (a tela, ou
uma imagem na memória). Não lê teclado nem controla o tempo — isso fica no
scripts/play.py. Separar assim permite reaproveitar o mesmo desenho para jogar,
assistir ao agente, gerar GIFs e o vídeo final.

Em pygame, uma Surface é como um <canvas> do HTML: você desenha retângulos e
círculos nela e depois mostra na janela.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Esconde a mensagem "Hello from the pygame community" que aparece no import.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from snake.env import DIRECTION_DELTAS, SnakeGame  # noqa: E402

Color = tuple[int, int, int]


@dataclass(frozen=True)
class RenderConfig:
    """Tamanhos e cores. Visual inspirado no Snake do Google, com identidade própria."""

    cell_size: int = 52
    border: int = 14          # moldura verde-escura (a "parede")
    hud_height: int = 64      # barra do placar, em cima
    light: Color = (170, 215, 81)
    dark: Color = (162, 209, 73)
    wall: Color = (74, 117, 44)
    obstacle: Color = (74, 117, 44)
    obstacle_edge: Color = (58, 94, 34)
    snake_head: Color = (78, 124, 246)
    snake_tail: Color = (36, 66, 158)
    fruit: Color = (231, 71, 29)
    leaf: Color = (56, 142, 60)
    text: Color = (255, 255, 255)
    overlay: tuple[int, int, int, int] = (0, 0, 0, 150)   # preto translúcido (RGBA)


class Renderer:
    def __init__(self, game: SnakeGame, config: RenderConfig | None = None):
        self.cfg = config or RenderConfig()
        self.rows = game.config.height
        self.cols = game.config.width
        cfg = self.cfg
        self.width = self.cols * cfg.cell_size + 2 * cfg.border
        self.height = self.rows * cfg.cell_size + 2 * cfg.border + cfg.hud_height
        if not pygame.font.get_init():
            pygame.font.init()
        # Font(None, ...) usa a fonte padrão do pygame (vem junto, funciona em qualquer PC).
        self.font_big = pygame.font.Font(None, 56)
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 26)

    # ------------------------------------------------------------------
    def cell_rect(self, cell: tuple[int, int]) -> pygame.Rect:
        """Retângulo (x, y, largura, altura) em pixels de uma célula (linha, coluna)."""
        r, c = cell
        s = self.cfg.cell_size
        return pygame.Rect(self.cfg.border + c * s, self.cfg.hud_height + self.cfg.border + r * s, s, s)

    def draw(self, surface: pygame.Surface, game: SnakeGame, hud_right: str = "") -> None:
        """Desenha o quadro inteiro: placar, tabuleiro, obstáculos, frutas e cobra."""
        cfg = self.cfg
        surface.fill(cfg.wall)
        self._draw_hud(surface, game, hud_right)

        # Tabuleiro quadriculado: (linha + coluna) par = claro, ímpar = escuro.
        for r in range(self.rows):
            for c in range(self.cols):
                color = cfg.light if (r + c) % 2 == 0 else cfg.dark
                pygame.draw.rect(surface, color, self.cell_rect((r, c)))

        for cell in game.obstacles:
            rect = self.cell_rect(cell)
            pygame.draw.rect(surface, cfg.obstacle, rect)
            pygame.draw.rect(surface, cfg.obstacle_edge, rect, width=3)

        for cell in game.fruits:
            self._draw_fruit(surface, cell)

        self._draw_snake(surface, game)

    def _draw_hud(self, surface, game, hud_right: str) -> None:
        cfg = self.cfg
        y = cfg.hud_height // 2
        # Ícone de fruta + contador, e ícone de obstáculo + contador.
        pygame.draw.circle(surface, cfg.fruit, (cfg.border + 18, y), 13)
        self._text(surface, str(game.fruits_collected), self.font, (cfg.border + 40, y), anchor="midleft")
        x = cfg.border + 120
        pygame.draw.rect(surface, cfg.obstacle_edge, (x, y - 12, 24, 24))
        pygame.draw.rect(surface, cfg.light, (x, y - 12, 24, 24), width=2)
        self._text(surface, str(game.obstacles_generated), self.font, (x + 34, y), anchor="midleft")
        if hud_right:
            self._text(surface, hud_right, self.font_small, (self.width - cfg.border, y), anchor="midright")

    def _draw_fruit(self, surface, cell) -> None:
        rect = self.cell_rect(cell)
        radius = int(self.cfg.cell_size * 0.34)
        center = (rect.centerx, rect.centery + 2)
        pygame.draw.circle(surface, self.cfg.fruit, center, radius)
        # Folhinha: uma elipse pequena em cima da fruta.
        leaf = pygame.Rect(0, 0, radius, radius // 2)
        leaf.midbottom = (rect.centerx + radius // 3, center[1] - radius + 4)
        pygame.draw.ellipse(surface, self.cfg.leaf, leaf)

    def _draw_snake(self, surface, game) -> None:
        cfg = self.cfg
        body = list(game.snake)
        n = len(body)
        inset = int(cfg.cell_size * 0.12)
        for i in range(n - 1, -1, -1):  # da cauda para a cabeça (a cabeça fica por cima)
            # Degradê: cabeça mais clara, cauda mais escura (interpolação linear).
            t = i / max(n - 1, 1)
            color = tuple(int(h + (tl - h) * t) for h, tl in zip(cfg.snake_head, cfg.snake_tail))
            rect = self.cell_rect(body[i]).inflate(-2 * inset, -2 * inset)
            pygame.draw.rect(surface, color, rect, border_radius=inset * 2)
            # "Ponte" até o próximo segmento, para a cobra parecer contínua.
            if i + 1 < n:
                a, b = self.cell_rect(body[i]), self.cell_rect(body[i + 1])
                bridge = a.union(b).inflate(-2 * inset, -2 * inset)
                pygame.draw.rect(surface, color, bridge)
        self._draw_eyes(surface, game)

    def _draw_eyes(self, surface, game) -> None:
        dr, dc = DIRECTION_DELTAS[game.direction]
        rect = self.cell_rect(game.head)
        s = self.cfg.cell_size
        cx, cy = rect.centerx + dc * s * 0.15, rect.centery + dr * s * 0.15
        # Os olhos ficam lado a lado, perpendiculares à direção do movimento.
        for side in (-1, 1):
            ex, ey = cx + side * dr * s * 0.18, cy + side * dc * s * 0.18
            pygame.draw.circle(surface, (255, 255, 255), (int(ex), int(ey)), int(s * 0.11))
            pygame.draw.circle(surface, (20, 20, 40), (int(ex + dc * 3), int(ey + dr * 3)), int(s * 0.05))

    # ------------------------------------------------------------------
    def draw_message(self, surface, title: str, lines: list[str]) -> None:
        """Escurece o tabuleiro e escreve uma mensagem no meio (início, pausa, fim)."""
        shade = pygame.Surface((self.width, self.height - self.cfg.hud_height), pygame.SRCALPHA)
        shade.fill(self.cfg.overlay)
        surface.blit(shade, (0, self.cfg.hud_height))
        cy = self.cfg.hud_height + (self.height - self.cfg.hud_height) // 2
        self._text(surface, title, self.font_big, (self.width // 2, cy - 20 * len(lines)))
        for i, line in enumerate(lines):
            self._text(surface, line, self.font_small, (self.width // 2, cy + 10 + 30 * i - 10 * len(lines)))

    def _text(self, surface, text, font, pos, anchor="center") -> None:
        image = font.render(text, True, self.cfg.text)
        rect = image.get_rect(**{anchor: pos})
        surface.blit(image, rect)
