"""Gráfico da Fase 5: curva de aprendizado de cada DQN contra as referências.

Lê experiments/<exp>/milestones.csv (avaliação rápida em 100 sementes oficiais
durante o treino) e desenha docs/media/fase5_dqn.png.

Como rodar:
    .venv\\Scripts\\python.exe scripts\\plot_dqn.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "media" / "fase5_dqn.png"

# (pasta do experimento, rótulo, cor). Cores fixas por série (paleta validada).
RUNS = [
    ("fase5_dqn_features", "DQN, 11 números", "#2a78d6"),
    ("fase5_dqn_space", "DQN, 18 números", "#eb6834"),
    ("fase5_dqn_cnn", "DQN, tabuleiro (CNN)", "#1baf7a"),
]
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"


def read_milestones(path: Path) -> tuple[list[float], list[float]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))  # cada linha vira um dicionário, como um objeto JS
    return [int(r["step"]) / 1e6 for r in rows], [float(r["eval_fruits_mean"]) for r in rows]


def main() -> None:
    greedy = json.loads((ROOT / "experiments" / "fase3_baselines" / "summary.json").read_text("utf-8"))
    human = json.loads((ROOT / "docs" / "human_baseline.json").read_text("utf-8"))
    refs = [("guloso", greedy["agents"]["greedy"]["fruits_mean"]), ("você", human["fruits_mean"])]

    fig, ax = plt.subplots(figsize=(9, 5))
    x_max = 0.0
    for folder, label, color in RUNS:
        path = ROOT / "experiments" / folder / "milestones.csv"
        if not path.exists():
            print(f"(sem {folder}, pulando)")
            continue
        xs, ys = read_milestones(path)
        x_max = max(x_max, xs[-1])
        ax.plot(xs, ys, color=color, linewidth=2, label=label, zorder=3)
        ax.text(xs[-1] * 1.01, ys[-1], f" {label}: {ys[-1]:.1f}", va="center", color=INK, fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=4)
    for name, value in refs:
        ax.axhline(value, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    for (name, value), va in zip(refs, ("bottom", "top")):  # guloso acima da linha, você abaixo
        ax.text(x_max * 1.34, value, f"{name} {value:.1f}", ha="right", va=va, color=MUTED, fontsize=9)

    ax.set_xlim(0, x_max * 1.35)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("passos de treino (milhões)", color=MUTED)
    ax.set_ylabel("frutas por partida (média em 100 sementes)", color=MUTED)
    ax.set_title("DQN aprendendo: avaliação durante o treino", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"Gráfico: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
