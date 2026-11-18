"""Treina um agente de Q-learning com tabela (Fase 4) e avalia nas sementes oficiais.

Como rodar (na raiz do projeto):
    .venv\\Scripts\\python.exe scripts\\train_qlearning.py --board mini --obs grid --episodes 1000000 --out fase4_q_grid_mini
    .venv\\Scripts\\python.exe scripts\\train_qlearning.py --board mini --obs features --episodes 20000 --out fase4_q_features_mini

Opções principais:
    --board     mini (5×5, sem obstáculos) ou full (jogo oficial)
    --obs       grid (tabuleiro inteiro = estado) ou features (resumo de 11 números)
    --episodes  quantas partidas de treino

Cada experimento vira uma pasta em experiments/<--out>/ (regra 4 do CLAUDE.md):
    config.json     todos os parâmetros + seed
    metrics.csv     curva de aprendizado (médias por bloco de partidas)
    milestones.csv  avaliação rápida (100 sementes) em marcos do treino
    summary.json    avaliação final nas 1.000 sementes oficiais
    best_replay.json  a melhor partida da avaliação final
    checkpoints/    a tabela Q em cada marco (não vai para o Git: pode ser grande)
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.qlearning import QLearningAgent  # noqa: E402
from snake import replay as rp  # noqa: E402
from snake.boards import BOARDS  # noqa: E402
from snake.evaluation import eval_seeds, evaluate, summarize  # noqa: E402
from snake.gym_env import Rewards, SnakeEnv  # noqa: E402


def epsilon_at(episode: int, args) -> float:
    """Exploração cai em linha reta de eps_start até eps_end e depois fica parada."""
    progress = min(1.0, episode / max(1, args.eps_decay * args.episodes))
    return args.eps_start + progress * (args.eps_end - args.eps_start)


def milestones(total: int) -> list[int]:
    """Marcos do treino: 0, 1 mil, 10 mil, 100 mil, ... e o fim (para checkpoints e vídeo)."""
    marks = [0] + [10**k for k in range(3, 8) if 10**k < total] + [total]
    return sorted(set(marks))


def quick_eval(agent: QLearningAgent, config, n: int) -> dict:
    agent.eval_steps = agent.eval_unseen = 0
    s = summarize(evaluate(agent, eval_seeds(n), config))
    s["unseen_rate"] = round(agent.eval_unseen / max(1, agent.eval_steps), 4)
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", choices=list(BOARDS), default="mini")
    parser.add_argument("--obs", choices=["grid", "features"], default="grid")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=float, default=0.8, help="fração do treino em que epsilon cai")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--out", required=True, help="nome da pasta em experiments/")
    args = parser.parse_args()

    out = ROOT / "experiments" / args.out
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = BOARDS[args.board]
    rewards = Rewards()
    (out / "config.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "rewards": asdict(rewards),
        "game_config": asdict(config),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    env = SnakeEnv(config, obs_type=args.obs, rewards=rewards)
    agent = QLearningAgent(args.obs, alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    # Sementes de TREINO vêm de um gerador próprio (nunca as de avaliação).
    seed_rng = np.random.default_rng(args.seed)
    marks = milestones(args.episodes)
    block = max(1, args.episodes // 500)  # ~500 pontos na curva de aprendizado

    metrics_file = (out / "metrics.csv").open("w", newline="", encoding="utf-8")
    metrics = csv.writer(metrics_file)
    metrics.writerow(["episode", "epsilon", "fruits_mean", "steps_mean", "return_mean", "table_size", "seconds"])
    miles_file = (out / "milestones.csv").open("w", newline="", encoding="utf-8")
    miles = csv.writer(miles_file)
    miles.writerow(["episode", "eval_fruits_mean", "eval_fruits_best", "eval_win_rate", "unseen_rate", "table_size"])

    def milestone(ep: int) -> None:
        s = quick_eval(agent, config, 100)
        miles.writerow([ep, s["fruits_mean"], s["fruits_best"], s["win_rate"], s["unseen_rate"], len(agent.q)])
        miles_file.flush()
        agent.save(out / "checkpoints" / f"q_ep{ep}.pkl")
        print(f"  marco {ep:>9,}: avaliação(100) {s['fruits_mean']:.2f} frutas | estados novos "
              f"{100 * s['unseen_rate']:.1f}% | tabela {len(agent.q):,}", flush=True)

    print(f"Treinando {args.obs}/{args.board} por {args.episodes:,} partidas -> experiments/{args.out}")
    start = time.perf_counter()
    milestone(0)
    fruits, steps, returns = [], [], []
    for ep in range(1, args.episodes + 1):
        epsilon = epsilon_at(ep, args)
        obs, _ = env.reset(seed=int(seed_rng.integers(0, 2**31 - 1)))
        total, done = 0.0, False
        while not done:
            action, _ = agent.choose(obs, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Truncada (tempo) NÃO é fim "de verdade": o futuro continua valendo.
            agent.learn(obs, action, reward, next_obs, terminated)
            obs, total, done = next_obs, total + reward, terminated or truncated
        fruits.append(info["fruits"])
        steps.append(info["steps"])
        returns.append(total)
        if ep % block == 0:
            metrics.writerow([ep, round(epsilon, 4), round(np.mean(fruits), 3), round(np.mean(steps), 1),
                              round(np.mean(returns), 3), len(agent.q), round(time.perf_counter() - start, 1)])
            fruits, steps, returns = [], [], []
        if ep in marks:
            milestone(ep)
    metrics_file.close()
    miles_file.close()
    train_seconds = time.perf_counter() - start

    print(f"Treino: {train_seconds:.0f} s. Avaliando nas {args.eval_games} sementes oficiais...")
    agent.eval_steps = agent.eval_unseen = 0
    results = evaluate(agent, eval_seeds(args.eval_games), config)
    summary = summarize(results)
    summary["unseen_rate"] = round(agent.eval_unseen / agent.eval_steps, 4)
    summary["table_size"] = len(agent.q)
    summary["train_seconds"] = round(train_seconds, 1)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best = max(results, key=lambda r: (r["fruits"], -r["steps"]))
    replay = rp.Replay(seed=best["seed"], actions=best["actions"], config=asdict(config), player=f"qlearning_{args.obs}")
    replay.result = rp.result_of(rp.simulate(replay))
    rp.save(replay, out / "best_replay.json")

    s = summary
    print(f"Avaliação final: média {s['fruits_mean']:.2f} | mediana {s['fruits_median']:g} | melhor {s['fruits_best']} "
          f"| vitórias {100 * s['win_rate']:.1f}% | estados nunca vistos {100 * s['unseen_rate']:.1f}% "
          f"| tabela {s['table_size']:,} estados")
    print("Fim: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in s["end_reasons"].items()))


if __name__ == "__main__":
    main()
