"""Estatísticas das suas partidas gravadas (replays/human/).

Cada replay é re-simulado do zero e conferido contra o resultado gravado, então
os números abaixo saem do motor do jogo, não de anotações.

Partidas OFICIAIS = as 20 primeiras jogadas com a regra atual (com a proibição de
becos sem saída, seção 4.5), em ordem cronológica. Elas são a referência humana
que os agentes vão tentar bater (M1 = média, M2 = recorde). As demais aparecem
como "extras" ou "regra antiga", só para histórico.

Como rodar:
    .venv\\Scripts\\python.exe scripts\\human_stats.py

Gera também:
    docs/media/fase2_humano.png      gráfico das partidas oficiais
    docs/human_baseline.json         resumo usado na tabela da Fase 3
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from snake import replay as rp  # noqa: E402

REPLAY_DIR = ROOT / "replays" / "human"
OFFICIAL_COUNT = 20
CHART_PATH = ROOT / "docs" / "media" / "fase2_humano.png"
BASELINE_PATH = ROOT / "docs" / "human_baseline.json"


def summarize(title: str, results: list[dict]) -> dict:
    """Imprime e devolve as métricas do CLAUDE.md para um grupo de partidas."""
    fruits = np.array([r["fruits"] for r in results])
    filled = np.array([r["board_filled"] for r in results])
    reasons = Counter(r["end_reason"] for r in results)
    n = len(results)
    print(f"\n== {title}: {n} partidas ==")
    print(f"Frutas: média {fruits.mean():.2f} | mediana {np.median(fruits):.1f} | recorde {fruits.max()}")
    print(f"Tabuleiro preenchido: média {100 * filled.mean():.1f}% | máx {100 * filled.max():.1f}%")
    print(f"Vitórias: {reasons.get('win', 0)} ({100 * reasons.get('win', 0) / n:.1f}%)")
    print("Fim de jogo: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
    # float()/int() porque o json não sabe gravar os tipos do numpy (np.float64...)
    return {
        "games": n,
        "fruits_mean": round(float(fruits.mean()), 2),
        "fruits_median": float(np.median(fruits)),
        "fruits_best": int(fruits.max()),
        "fruits_std": round(float(fruits.std()), 2),
        "board_filled_mean": round(float(filled.mean()), 4),
        "board_filled_max": round(float(filled.max()), 4),
        "win_rate": reasons.get("win", 0) / n,
        "end_reasons": dict(reasons),
    }


def plot(fruits: list[int], path: Path) -> None:
    """Barras com as frutas de cada partida oficial, na ordem em que foram jogadas."""
    import matplotlib

    matplotlib.use("Agg")  # gera a imagem sem abrir janela
    import matplotlib.pyplot as plt

    ink, muted, grid, bar = "#1f2328", "#59636e", "#d8dee4", "#2f6fdb"
    mean, median, best = np.mean(fruits), np.median(fruits), max(fruits)
    games = np.arange(1, len(fruits) + 1)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(games, fruits, width=0.7, color=bar, zorder=2)
    ax.hlines(mean, 0.5, len(fruits) + 0.5, color=ink, linewidth=1.5, linestyle="--", zorder=3)
    ax.text(len(fruits) + 0.6, mean, f"média {mean:.1f}", va="center", color=ink, fontsize=9)
    ax.hlines(median, 0.5, len(fruits) + 0.5, color=muted, linewidth=1, linestyle=":", zorder=3)
    ax.text(len(fruits) + 0.6, median, f"mediana {median:.0f}", va="center", color=muted, fontsize=9)
    i = fruits.index(best)
    ax.text(games[i], best + 0.6, f"recorde {best}", ha="center", color=ink, fontsize=9)

    ax.set_title("Minhas 20 partidas oficiais (regra atual)", loc="left", color=ink, fontsize=12)
    ax.set_xlabel("partida (ordem em que joguei)", color=muted)
    ax.set_ylabel("frutas", color=muted)
    ax.set_xticks(games)
    ax.set_xlim(0.3, len(fruits) + 3.2)
    ax.set_ylim(0, best + 4)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nGráfico salvo em {path.relative_to(ROOT)}")


def main() -> None:
    paths = sorted(REPLAY_DIR.glob("*.json"))  # o nome começa pela data: ordem cronológica
    if not paths:
        print("Nenhuma partida gravada ainda. Jogue com scripts\\play.py.")
        return

    groups: dict[str, list[dict]] = {"oficial": [], "extra": [], "antiga": []}
    official_files: list[str] = []
    print(f"{'#':>3}  {'arquivo':<36} {'grupo':<7} {'frutas':>6} {'obst.':>5} {'passos':>6}  fim")
    for i, path in enumerate(paths, 1):
        replay = rp.load(path)
        rp.simulate(replay)  # levanta erro se o replay não reproduzir o resultado gravado
        if not replay.game_config().forbid_dead_ends:
            group = "antiga"
        elif len(groups["oficial"]) < OFFICIAL_COUNT:
            group = "oficial"
            official_files.append(path.name)
        else:
            group = "extra"
        r = replay.result
        groups[group].append(r)
        print(f"{i:>3}  {path.name:<36} {group:<7} {r['fruits']:>6} {r['obstacles']:>5} {r['steps']:>6}  {r['end_reason']}")

    print(f"\nTodas as {len(paths)} partidas foram re-simuladas e conferidas.")
    if groups["antiga"]:
        summarize("Regra antiga (sem a proibição de becos; só histórico)", groups["antiga"])
    if groups["extra"]:
        summarize("Extras com a regra atual (depois das 20 oficiais)", groups["extra"])
    if len(groups["oficial"]) < OFFICIAL_COUNT:
        print(f"\nPartidas oficiais: {len(groups['oficial'])} de {OFFICIAL_COUNT}. Jogue mais "
              f"{OFFICIAL_COUNT - len(groups['oficial'])} com scripts\\play.py.")
        return

    stats = summarize("OFICIAIS (referência humana para os agentes)", groups["oficial"])
    plot([r["fruits"] for r in groups["oficial"]], CHART_PATH)
    BASELINE_PATH.write_text(
        json.dumps({**stats, "replays": official_files}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Resumo salvo em {BASELINE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
