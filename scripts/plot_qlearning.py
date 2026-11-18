"""Gráfico da Fase 4: curvas de aprendizado e tamanho da tabela dos experimentos de Q-learning.

Como rodar:
    .venv\\Scripts\\python.exe scripts\\plot_qlearning.py

Lê experiments/fase4_q_*/metrics.csv e gera docs/media/fase4_qlearning.png com 2 painéis:
    esquerda: frutas por partida durante o treino (média de cada bloco)
    direita:  quantos estados a tabela já guardou
O eixo x (partidas de treino) é logarítmico: 1 mil, 10 mil, 100 mil, 1 milhão
ficam igualmente espaçados, porque os experimentos têm tamanhos muito diferentes.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPERIMENTS = [  # (pasta, rótulo, cor) — cores validadas para daltonismo
    ("fase4_q_grid_mini", "tabuleiro inteiro · 5×5", "#2a78d6"),
    ("fase4_q_features_mini", "resumo de 11 números · 5×5", "#eb6834"),
    ("fase4_q_features_full", "resumo de 11 números · jogo oficial", "#1baf7a"),
    ("fase4_q_grid_full", "tabuleiro inteiro · jogo oficial", "#eda100"),
]
OUT = ROOT / "docs" / "media" / "fase4_qlearning.png"


def read_metrics(folder: str) -> dict[str, list[float]]:
    with (ROOT / "experiments" / folder / "metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key: [float(r[key]) for r in rows] for key in rows[0]}


def smooth(values: list[float], window: int = 10) -> list[float]:
    """Média móvel: cada ponto vira a média dele com os vizinhos anteriores (suaviza o ruído)."""
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2328", "#59636e", "#d8dee4"
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))
    for folder, label, color in EXPERIMENTS:
        m = read_metrics(folder)
        y = smooth(m["fruits_mean"])
        left.plot(m["episode"], y, color=color, linewidth=2, label=label)
        left.text(m["episode"][-1] * 1.15, y[-1], f"{y[-1]:.1f}", color=ink, va="center", fontsize=9)
        right.plot(m["episode"], m["table_size"], color=color, linewidth=2, label=label)
        right.text(m["episode"][-1] * 1.15, m["table_size"][-1], f"{int(m['table_size'][-1]):,}".replace(",", "."),
                   color=ink, va="center", fontsize=9)

    left.set_title("Frutas por partida no treino", loc="left", color=ink)
    left.set_ylabel("frutas (média do bloco)", color=muted)
    right.set_title("Estados guardados na tabela", loc="left", color=ink)
    right.set_yscale("log")
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xlabel("partidas de treino (escala log)", color=muted)
        ax.grid(color=grid, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)
        ax.set_xlim(right=ax.get_xlim()[1] * 3)  # espaço para os rótulos no fim das linhas
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"Gráfico salvo em {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
