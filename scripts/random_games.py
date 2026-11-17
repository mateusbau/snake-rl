"""Roda milhares de partidas aleatórias para testar o motor do jogo (Fase 1).

A cada passo de cada partida, confere todas as invariantes (snake/checks.py).
Se algo quebrar, o script para e mostra a seed da partida com problema.

Dois "jogadores" bobos, só para exercitar o motor (não são os baselines da Fase 3):
- random:      sorteia qualquer uma das 4 direções;
- safe_random: sorteia só entre as direções que não matam no passo seguinte,
               para as partidas durarem mais e encherem o mapa de obstáculos.

Como rodar (na raiz do projeto):
    .venv\\Scripts\\python.exe scripts\\random_games.py --games 10000
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Permite rodar o script direto (python scripts/x.py) e ainda importar o pacote `snake`,
# que está na pasta de cima.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from snake.checks import check_invariants  # noqa: E402
from snake.env import DOWN, LEFT, NEIGHBORS_8, RIGHT, UP, SnakeGame  # noqa: E402

ALL_DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


def play(game_seed: int, player: str, rng: np.random.Generator) -> dict:
    """Joga uma partida inteira e devolve as estatísticas dela."""
    game = SnakeGame(seed=game_seed)
    check_invariants(game)
    while not game.game_over:
        if player == "safe_random":
            options = game.safe_directions() or [game.direction]
        else:
            options = ALL_DIRECTIONS
        game.step(int(rng.choice(options)))
        try:
            check_invariants(game)
        except AssertionError as error:
            print(f"\nInvariante quebrada! jogador={player} seed={game_seed}: {error}")
            print(game)
            raise
    return {
        "fruits": game.fruits_collected,
        "obstacles": game.obstacles_generated,
        "steps": game.steps,
        "end_reason": game.end_reason,
    }


def summarize(name: str, games: list[dict], seconds: float) -> None:
    fruits = np.array([g["fruits"] for g in games])
    obstacles = np.array([g["obstacles"] for g in games])
    steps = np.array([g["steps"] for g in games])
    reasons = Counter(g["end_reason"] for g in games)
    print(f"\n== {name}: {len(games)} partidas em {seconds:.1f} s "
          f"({sum(steps) / seconds:,.0f} passos/s) ==")
    print(f"frutas      média {fruits.mean():6.2f} | mediana {np.median(fruits):4.0f} | máx {fruits.max()}")
    print(f"obstáculos  média {obstacles.mean():6.2f} | máx {obstacles.max()}")
    print(f"passos      média {steps.mean():6.1f} | máx {steps.max()}")
    print("fim de jogo: " + ", ".join(f"{r} {100 * n / len(games):.1f}%" for r, n in reasons.most_common()))


def saturation_counts(n: int) -> list[int]:
    """Enche o mapa inicial de obstáculos até não caber mais nenhum, n vezes."""
    counts = []
    for seed in range(n):
        game = SnakeGame(seed=seed)
        while game.try_spawn_obstacle() is not None:
            pass
        # Aqui os obstáculos não vêm de frutas comidas, então só a regra de
        # vizinhança (8 direções) faz sentido conferir.
        obstacles = set(game.obstacles)
        for r, c in obstacles:
            assert not any((r + dr, c + dc) in obstacles for dr, dc in NEIGHBORS_8)
        counts.append(len(game.obstacles))
    return counts


def plot(results: dict[str, list[dict]], saturation: list[int], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # gera a imagem sem abrir janela
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"random": "#9aa5b1", "safe_random": "#2f6fd6"}

    # Esquerda: saturação — quantos obstáculos cabem no mapa inicial, sorteando
    # um atrás do outro até não sobrar posição válida.
    ax = axes[0]
    counts = Counter(saturation)
    xs = sorted(counts)
    ax.bar(xs, [counts[x] for x in xs], color=colors["safe_random"])
    ax.axvline(25, ls="--", color="#555")
    ax.text(24.7, max(counts.values()) * 0.95, "máximo teórico: 25", ha="right", color="#555")
    ax.set_xticks(range(min(xs), 26))
    ax.set_xlabel("obstáculos até não caber mais nenhum")
    ax.set_ylabel("sorteios")
    ax.set_title(f"Saturação do mapa ({len(saturation):,} sorteios)".replace(",", "."))

    # Direita: distribuição de frutas por partida, para os dois jogadores.
    ax = axes[1]
    top = max(g["fruits"] for g in results["safe_random"])
    bins = np.arange(0, top + 2) - 0.5
    for name, games in results.items():
        ax.hist([g["fruits"] for g in games], bins=bins, alpha=0.7, color=colors[name], label=name)
    ax.set_xlabel("frutas por partida")
    ax.set_ylabel("partidas")
    ax.set_title(f"Jogadores aleatórios ({len(games):,} partidas cada)".replace(",", "."))
    ax.legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"\nGráfico salvo em {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=10_000, help="partidas por jogador")
    parser.add_argument("--seed", type=int, default=0, help="seed das escolhas dos jogadores")
    parser.add_argument("--plot", type=Path, default=ROOT / "docs" / "media" / "fase1_partidas_aleatorias.png")
    args = parser.parse_args()

    results = {}
    for player in ("random", "safe_random"):
        rng = np.random.default_rng(args.seed)
        start = time.perf_counter()
        results[player] = [play(game_seed, player, rng) for game_seed in range(args.games)]
        summarize(player, results[player], time.perf_counter() - start)

    print(f"\nNenhuma invariante quebrou em {2 * args.games} partidas.")

    saturation = saturation_counts(args.games)
    sat = np.array(saturation)
    print(f"Saturação: {sat.mean():.1f} obstáculos em média até lotar o mapa (mín {sat.min()}, máx {sat.max()})")
    plot(results, saturation, args.plot)


if __name__ == "__main__":
    main()
