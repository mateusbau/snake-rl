"""Avalia agentes nas 1.000 sementes oficiais e compara com a referência humana.

Toda comparação do projeto sai daqui (regra 3 do CLAUDE.md): mesmas sementes,
mesmas métricas, para qualquer agente.

Como rodar (na raiz do projeto):
    .venv\\Scripts\\python.exe scripts\\evaluate.py                         # os 4 agentes da Fase 3
    .venv\\Scripts\\python.exe scripts\\evaluate.py --agents greedy bfs     # só alguns
    .venv\\Scripts\\python.exe scripts\\evaluate.py --games 100             # teste rápido (100 primeiras sementes)

Saídas em experiments/<--out>/ (padrão: fase3_baselines):
    summary.json            métricas de cada agente + metadados da avaliação
    <agente>_games.csv      uma linha por partida (para gráficos)
    replays/<agente>_best.json   a melhor partida de cada agente (assista com play.py --replay)
    tabela.md               tabela comparativa (agentes + você)
e o gráfico docs/media/fase3_comparacao.png.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.simple import AGENTS  # noqa: E402
from snake import replay as rp  # noqa: E402
from snake.evaluation import EVAL_MASTER_SEED, eval_seeds, evaluate, summarize  # noqa: E402

HUMAN_PATH = ROOT / "docs" / "human_baseline.json"
CHART_PATH = ROOT / "docs" / "media" / "fase3_comparacao.png"
REASON_NAMES = {"wall": "parede", "body": "corpo", "obstacle": "obstáculo", "timeout": "tempo", "win": "vitória"}
LABELS = {"random": "Aleatório", "safe_random": "Aleatório seguro", "greedy": "Guloso",
          "bfs": "BFS + flood fill", "human": "Eu (humano)"}


def seeds_fingerprint(seeds: list[int]) -> str:
    """Uma "impressão digital" curta das sementes, para provar que a prova foi a mesma."""
    return hashlib.sha256(",".join(map(str, seeds)).encode()).hexdigest()[:12]


def save_games_csv(results: list[dict], path: Path) -> None:
    columns = ["seed", "fruits", "steps", "obstacles", "length", "board_filled", "end_reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")  # ignora "actions"
        writer.writeheader()
        writer.writerows(results)


def save_best_replay(name: str, results: list[dict], path: Path) -> dict:
    """Guarda a melhor partida (mais frutas; empate -> menos passos) como replay."""
    best = max(results, key=lambda r: (r["fruits"], -r["steps"]))
    replay = rp.Replay(seed=best["seed"], actions=best["actions"], player=name)
    replay.result = rp.result_of(rp.simulate(replay))  # re-simula: confere que reproduz
    rp.save(replay, path)
    return best


def format_reasons(reasons: dict) -> str:
    return ", ".join(f"{REASON_NAMES.get(k, k)} {100 * v:.0f}%" for k, v in reasons.items())


def markdown_table(summaries: dict[str, dict], human: dict | None) -> str:
    rows = [
        "| Jogador | Partidas | Média | Mediana | Melhor | % tabuleiro (média / máx) | Vitórias | Como terminou |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    entries = list(summaries.items())
    if human:
        human = {**human, "end_reasons": {k: v / human["games"] for k, v in human["end_reasons"].items()}}
        entries.append(("human", human))
    for name, s in sorted(entries, key=lambda e: e[1]["fruits_mean"]):
        mean = f"{s['fruits_mean']:.2f}"
        if "fruits_sem" in s:
            mean += f" ± {2 * s['fruits_sem']:.2f}"
        rows.append(
            f"| {LABELS.get(name, name)} | {s['games']} | {mean} | {s['fruits_median']:g} | {s['fruits_best']} "
            f"| {100 * s['board_filled_mean']:.1f}% / {100 * s['board_filled_max']:.1f}% "
            f"| {100 * s['win_rate']:.1f}% | {format_reasons(s['end_reasons'])} |"
        )
    note = ("\n± = 2 erros padrão (intervalo de ~95% para a média). Agentes: as mesmas "
            "1.000 sementes. Humano: 20 partidas oficiais da Fase 2 (outras sementes).\n")
    return "\n".join(rows) + "\n" + note


def plot(summaries: dict[str, dict], human: dict | None, path: Path) -> None:
    """Barras horizontais: média de frutas de cada jogador (o humano em destaque)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, bar, accent = "#1f2328", "#59636e", "#d8dee4", "#2f6fdb", "#e8590c"
    entries = [(name, s["fruits_mean"], s["fruits_best"]) for name, s in summaries.items()]
    if human:
        entries.append(("human", human["fruits_mean"], human["fruits_best"]))
    entries.sort(key=lambda e: e[1])

    fig, ax = plt.subplots(figsize=(9, 0.75 * len(entries) + 1.4))
    y = range(len(entries))
    colors = [accent if name == "human" else bar for name, _, _ in entries]
    ax.barh(list(y), [e[1] for e in entries], height=0.6, color=colors, zorder=2)
    top = max(e[1] for e in entries)
    for i, (name, mean, best) in enumerate(entries):
        ax.text(mean + top * 0.015, i, f"{mean:.1f}  (melhor {best})", va="center", color=ink, fontsize=10)
    ax.set_yticks(list(y))
    ax.set_yticklabels([LABELS.get(e[0], e[0]) for e in entries], color=ink)
    ax.set_xlim(0, top * 1.3)
    ax.set_xlabel("frutas por partida (média)", color=muted)
    ax.set_title("Agentes sem ML x humano", loc="left", color=ink, fontsize=12)
    ax.grid(axis="x", color=grid, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", default=list(AGENTS), choices=list(AGENTS))
    parser.add_argument("--games", type=int, default=1000, help="usa as N primeiras sementes oficiais")
    parser.add_argument("--out", default="fase3_baselines", help="pasta dentro de experiments/")
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args()

    seeds = eval_seeds(args.games)
    out = ROOT / "experiments" / args.out
    (out / "replays").mkdir(parents=True, exist_ok=True)
    print(f"Avaliando {len(seeds)} partidas por agente (sementes {seeds_fingerprint(seeds)})")

    summaries, seconds = {}, {}
    for name in args.agents:
        start = time.perf_counter()
        results = evaluate(AGENTS[name](), seeds, progress=True)
        seconds[name] = round(time.perf_counter() - start, 1)
        summaries[name] = summarize(results)
        save_games_csv(results, out / f"{name}_games.csv")
        best = save_best_replay(name, results, out / "replays" / f"{name}_best.json")
        s = summaries[name]
        print(f"{name:>12}: média {s['fruits_mean']:.2f} | mediana {s['fruits_median']:g} | melhor {s['fruits_best']} "
              f"(seed {best['seed']}) | vitórias {100 * s['win_rate']:.1f}% | {seconds[name]} s")

    human = json.loads(HUMAN_PATH.read_text(encoding="utf-8")) if HUMAN_PATH.exists() else None
    table = markdown_table(summaries, human)
    (out / "tabela.md").write_text(table, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "eval_master_seed": EVAL_MASTER_SEED,
        "games": len(seeds),
        "seeds_fingerprint": seeds_fingerprint(seeds),
        "seconds": seconds,
        "agents": summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_chart:
        plot(summaries, human, CHART_PATH)
        print(f"Gráfico: {CHART_PATH.relative_to(ROOT)}")
    print("\n" + table)
    print(f"Resultados em {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
