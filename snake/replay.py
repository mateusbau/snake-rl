"""Replays: guardar uma partida como "seed + lista de ações" e reproduzi-la depois.

Por que isso funciona? O motor é determinístico: com a mesma seed e as mesmas
ações, a partida sai idêntica, passo a passo. Então não precisamos gravar vídeo
nem o tabuleiro inteiro a cada passo — só o que o jogador apertou. Um replay de
200 passos ocupa poucos KB e pode ser re-renderizado depois em qualquer
resolução (útil para o vídeo do LinkedIn).

Formato do arquivo (JSON, legível em qualquer editor):
    {
      "seed": 123,
      "player": "human",
      "actions": [1, null, null, 0, ...],   # direção por passo; null = manteve
      "config": {...},                      # GameConfig usada
      "result": {"fruits": 12, "steps": 180, "end_reason": "body", ...},
      "created_at": "2026-11-17T21:30:00"
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from snake.env import GameConfig, SnakeGame


@dataclass
class Replay:
    seed: int
    actions: list[int | None] = field(default_factory=list)
    config: dict = field(default_factory=lambda: asdict(GameConfig()))
    player: str = "human"
    result: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def game_config(self) -> GameConfig:
        """Recria a GameConfig. O JSON transforma tuplas em listas, então
        convertemos de volta (listas de listas -> tuplas de tuplas)."""
        cfg = {
            key: tuple(tuple(v) if isinstance(v, list) else v for v in value) if isinstance(value, list) else value
            for key, value in self.config.items()
        }
        # Replays gravados antes da regra 4.5 não têm essa chave: foram jogados
        # SEM a regra, então precisam ser reproduzidos sem ela.
        cfg.setdefault("forbid_dead_ends", False)
        return GameConfig(**cfg)


def result_of(game: SnakeGame) -> dict:
    """Resumo do fim de uma partida (vai dentro do replay)."""
    cells = game.config.height * game.config.width
    return {
        "fruits": game.fruits_collected,
        "steps": game.steps,
        "obstacles": game.obstacles_generated,
        "length": len(game.snake),
        # % do tabuleiro preenchido pela cobra (uma das métricas do CLAUDE.md)
        "board_filled": round(len(game.snake) / cells, 4),
        "end_reason": game.end_reason,
    }


def save(replay: Replay, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(replay)
    # Ações numa linha só, para o arquivo ficar curto e legível: trocamos a lista
    # por um marcador, formatamos o resto com indentação e depois devolvemos a lista.
    actions = data.pop("actions")
    config = data.pop("config")
    data = {"seed": data.pop("seed"), "actions": "__ACTIONS__", "config": "__CONFIG__", **data}
    text = json.dumps(data, ensure_ascii=False, indent=2)
    text = text.replace('"__ACTIONS__"', json.dumps(actions)).replace('"__CONFIG__"', json.dumps(config))
    path.write_text(text, encoding="utf-8")


def load(path: Path) -> Replay:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Replay(**data)


def frames(replay: Replay) -> Iterator[SnakeGame]:
    """Reproduz o replay, entregando o jogo no início e depois de cada passo.

    Atenção: é sempre o MESMO objeto `game` sendo atualizado (para desenhar na
    hora). Se precisar guardar estados, copie (ex.: str(game) ou game.get_state()).
    """
    game = SnakeGame(replay.game_config(), seed=replay.seed)
    yield game
    for action in replay.actions:
        game.step(action)
        yield game


def simulate(replay: Replay) -> SnakeGame:
    """Roda o replay inteiro e confere se o resultado bate com o gravado."""
    game = None
    for game in frames(replay):
        pass
    if replay.result:
        assert result_of(game) == replay.result, (
            f"Replay não bate com o resultado gravado: {result_of(game)} != {replay.result}"
        )
    return game
