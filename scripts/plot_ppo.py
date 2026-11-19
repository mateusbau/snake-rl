"""Gráfico da Fase 6: curvas de aprendizado dos PPOs contra as referências.

Lê experiments/<exp>/milestones.csv e desenha docs/media/fase6_ppo.png.

Como rodar:
    .venv\\Scripts\\python.exe scripts\\plot_ppo.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_dqn import GRID, INK, MUTED, read_milestones  # noqa: E402  (mesma pasta scripts/)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "media" / "fase6_ppo.png"
RUNS = [
    ("fase6_ppo", "PPO", "#2a78d6"),
    ("fase6_maskppo", "MaskablePPO", "#eb6834"),
    ("fase6_ppo_cur", "PPO + currículo", "#1baf7a"),
    ("fase6_maskppo_ent", "MaskablePPO, mais exploração", "#eda100"),
]


def main() -> None:
    comp = json.loads((ROOT / "experiments" / "fase5_comparacao" / "summary.json").read_text("utf-8"))["agents"]
    human = json.loads((ROOT / "docs" / "human_baseline.json").read_text("utf-8"))
    refs = [("BFS", comp["bfs"]["fruits_mean"]), ("DQN 18 números", comp["dqn_18"]["fruits_mean"]),
            ("você", human["fruits_mean"])]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x_max = 0.0
    ends = []
    for folder, label, color in RUNS:
        path = ROOT / "experiments" / folder / "milestones.csv"
        if not path.exists():
            continue
        xs, ys = read_milestones(path)
        x_max = max(x_max, xs[-1])
        ax.plot(xs, ys, color=color, linewidth=2, label=label, zorder=3)
        ends.append((ys[-1], xs[-1], label))
    # Rótulos na ponta das curvas, afastados para não se sobreporem.
    ends.sort()
    last_y = -10.0
    for y, x, label in ends:
        y_text = max(y, last_y + 2.6)
        ax.text(x * 1.01, y_text, f" {label}: {y:.1f}", va="center", color=INK, fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=4)
        last_y = y_text
    for name, value in refs:
        ax.axhline(value, color=MUTED, linewidth=1, linestyle="--", zorder=1)
        ax.text(x_max * 0.01, value, f"{name} {value:.1f}", ha="left", va="bottom", color=MUTED, fontsize=9)

    ax.set_xlim(0, x_max * 1.43)
    ax.set_ylim(0, 60)
    ax.set_xlabel("passos de treino (milhões, somando os 8 tabuleiros)", color=MUTED)
    ax.set_ylabel("frutas por partida (média em 100 sementes)", color=MUTED)
    ax.set_title("PPO aprendendo: avaliação durante o treino", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"Gráfico: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
