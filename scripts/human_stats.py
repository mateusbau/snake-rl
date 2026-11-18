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


def summarize(title: str, results: list[dict]) -> None:
    fruits = np.array([r["fruits"] for r in results])
    filled = np.array([r["board_filled"] for r in results])
    reasons = Counter(r["end_reason"] for r in results)
    print(f"\n== {title}: {len(results)} partidas ==")
    print(f"Frutas: média {fruits.mean():.2f} | mediana {np.median(fruits):.1f} | recorde {fruits.max()}")
    print(f"Tabuleiro preenchido: média {100 * filled.mean():.1f}% | máx {100 * filled.max():.1f}%")
    print(f"Vitórias: {reasons.get('win', 0)} ({100 * reasons.get('win', 0) / len(results):.1f}%)")
    print("Fim de jogo: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))


def main() -> None:
    paths = sorted(REPLAY_DIR.glob("*.json"))
    if not paths:
        print("Nenhuma partida gravada ainda. Jogue com scripts\\play.py.")
        return

    # As partidas são separadas pela regra com que foram jogadas: antes ou depois
    # da proibição de becos sem saída (seção 4.5). Só as da regra atual valem para
    # a comparação com os agentes.
    groups: dict[str, list[dict]] = {"atual": [], "antiga": []}
    print(f"{'#':>3}  {'arquivo':<36} {'regra':<7} {'frutas':>6} {'obst.':>5} {'passos':>6}  fim")
    for i, path in enumerate(paths, 1):
        replay = rp.load(path)
        rp.simulate(replay)  # levanta erro se o replay não reproduzir o resultado gravado
        rule = "atual" if replay.game_config().forbid_dead_ends else "antiga"
        r = replay.result
        groups[rule].append(r)
        print(f"{i:>3}  {path.name:<36} {rule:<7} {r['fruits']:>6} {r['obstacles']:>5} {r['steps']:>6}  {r['end_reason']}")

    print(f"\nTodas as {len(paths)} partidas foram re-simuladas e conferidas.")
    if groups["antiga"]:
        summarize("Regra antiga (sem a proibição de becos, seção 4.5)", groups["antiga"])
    if groups["atual"]:
        summarize("Regra atual (valem para a comparação com os agentes)", groups["atual"])
    else:
        print("\nAinda não há partidas com a regra atual.")


if __name__ == "__main__":
    main()
