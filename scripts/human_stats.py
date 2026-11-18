"""Estatísticas das suas partidas gravadas (replays/human/).

Cada replay é re-simulado do zero e conferido contra o resultado gravado, então
os números abaixo saem do motor do jogo, não de anotações.

Como rodar:
    .venv\\Scripts\\python.exe scripts\\human_stats.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from snake import replay as rp  # noqa: E402

REPLAY_DIR = ROOT / "replays" / "human"


def main() -> None:
    paths = sorted(REPLAY_DIR.glob("*.json"))
    if not paths:
        print("Nenhuma partida gravada ainda. Jogue com scripts\\play.py.")
        return

    results = []
    print(f"{'#':>3}  {'arquivo':<34} {'frutas':>6} {'obst.':>5} {'passos':>6}  fim")
    for i, path in enumerate(paths, 1):
        replay = rp.load(path)
        rp.simulate(replay)  # levanta erro se o replay não reproduzir o resultado gravado
        r = replay.result
        results.append(r)
        print(f"{i:>3}  {path.name:<34} {r['fruits']:>6} {r['obstacles']:>5} {r['steps']:>6}  {r['end_reason']}")

    fruits = np.array([r["fruits"] for r in results])
    filled = np.array([r["board_filled"] for r in results])
    reasons = Counter(r["end_reason"] for r in results)
    print(f"\nPartidas: {len(results)} (todas re-simuladas e conferidas)")
    print(f"Frutas: média {fruits.mean():.2f} | mediana {np.median(fruits):.1f} | recorde {fruits.max()}")
    print(f"Tabuleiro preenchido: média {100 * filled.mean():.1f}% | máx {100 * filled.max():.1f}%")
    print(f"Vitórias: {reasons.get('win', 0)} ({100 * reasons.get('win', 0) / len(results):.1f}%)")
    print("Fim de jogo: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))


if __name__ == "__main__":
    main()
