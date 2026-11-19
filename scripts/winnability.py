"""Fase 7: zerar é sequer POSSÍVEL? Uma conta de paridade responde quase sempre.

A ideia (com um tabuleiro de xadrez): pinte as células alternando preto e branco,
cor = (linha + coluna) % 2. A cobra anda uma casa por vez, então segmentos
vizinhos do corpo SEMPRE têm cores diferentes: o corpo alterna preto, branco,
preto, branco...

Para zerar, a cobra tem que cobrir TODAS as células livres (as sem obstáculo).
Uma fila que alterna cores tem no máximo 1 célula a mais de uma cor que da outra.
Então:

    se |livres pretas - livres brancas| >= 2  ->  zerar é IMPOSSÍVEL, jogue como jogar.

Exemplo: 9×10 = 45 pretas + 45 brancas. Com 16 obstáculos, 11 em pretas e 5 em
brancas: livres = 34 pretas + 40 brancas -> diferença 6 -> nenhuma cobra zera esse mapa.

O script joga um agente nas sementes oficiais, guarda os obstáculos do fim de cada
partida e conta em quantas zerar ainda era possível por esse critério (é uma
condição NECESSÁRIA: passar nela não garante que dá para zerar).

Como rodar:
    .venv\\Scripts\\python.exe scripts\\winnability.py --agent bfs
    .venv\\Scripts\\python.exe scripts\\winnability.py --model experiments/fase6_ppo/model.zip --games 1000
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.search import color_imbalance  # noqa: E402
from agents.simple import AGENTS  # noqa: E402
from snake.env import SnakeGame  # noqa: E402
from snake.evaluation import eval_seeds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=list(AGENTS), default="bfs")
    parser.add_argument("--model", help="rede treinada (.pt ou .zip) em vez de --agent")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--out", help="salva o resumo em JSON neste caminho")
    args = parser.parse_args()

    if args.model:
        sys.path.insert(0, str(ROOT / "scripts"))
        from evaluate import load_model

        name, agent = load_model(f"model={args.model}")
        name = args.model
    else:
        name, agent = args.agent, AGENTS[args.agent]()

    imbalances, saturated_late = Counter(), 0
    for seed in eval_seeds(args.games):
        game = SnakeGame(seed=seed)
        agent.reset(seed)
        while not game.game_over:
            game.step(agent.act(game))
        imbalances[color_imbalance(game.obstacles, game.config.height, game.config.width)] += 1
        saturated_late += not game.empty_cells()  # terminou sem nenhuma célula vazia (fim de jogo de verdade)

    n = args.games
    possible = sum(v for k, v in imbalances.items() if k <= 1)
    print(f"Agente: {name} | {n} partidas")
    print(f"Diferença de cores nas células livres (no fim): {dict(sorted(imbalances.items()))}")
    print(f"Zerar ainda era possível (diferença <= 1): {possible} partidas ({100 * possible / n:.1f}%)")
    print(f"Partidas que terminaram sem nenhuma célula vazia: {saturated_late}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "agent": name, "games": n, "imbalance_counts": dict(sorted(imbalances.items())),
            "parity_possible": possible, "ended_without_empty_cells": saturated_late,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
