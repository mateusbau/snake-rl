"""Junta avaliações feitas em pastas separadas numa tabela e num gráfico só.

Útil quando agentes lentos (com busca) são avaliados em processos paralelos, cada
um com seu --out. Só junta avaliações com a MESMA impressão digital de sementes.

Como rodar:
    .venv\\Scripts\\python.exe scripts\\merge_results.py --inputs fase6_comparacao fase7_eval_outros \\
        --only greedy bfs ppo ppo_seg --out fase7_comparacao --chart docs\\media\\fase7_comparacao.png --title "..."
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import HUMAN_PATH, markdown_table, plot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="pastas em experiments/ com summary.json")
    parser.add_argument("--only", nargs="*", help="só estes agentes (na ordem que quiser)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--chart", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    agents, fingerprints = {}, set()
    for folder in args.inputs:
        data = json.loads((ROOT / "experiments" / folder / "summary.json").read_text(encoding="utf-8"))
        fingerprints.add(data["seeds_fingerprint"])
        agents.update(data["agents"])
    if len(fingerprints) != 1:
        raise SystemExit(f"Sementes diferentes entre as avaliações: {fingerprints}")
    if args.only:
        agents = {name: agents[name] for name in args.only}

    human = json.loads(HUMAN_PATH.read_text(encoding="utf-8"))
    out = ROOT / "experiments" / args.out
    out.mkdir(parents=True, exist_ok=True)
    table = markdown_table(agents, human)
    (out / "tabela.md").write_text(table, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({
        "merged_from": args.inputs, "seeds_fingerprint": fingerprints.pop(), "agents": agents,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(agents, human, ROOT / args.chart, args.title)
    print(table)


if __name__ == "__main__":
    main()
