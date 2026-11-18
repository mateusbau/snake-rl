"""Treina um DQN (Fase 5) e avalia nas 1.000 sementes oficiais.

Como rodar (na raiz do projeto):
    .venv\\Scripts\\python.exe scripts\\train_dqn.py --obs features --steps 300000 --out fase5_dqn_features
    .venv\\Scripts\\python.exe scripts\\train_dqn.py --obs features_space --steps 1000000 --out fase5_dqn_space
    .venv\\Scripts\\python.exe scripts\\train_dqn.py --obs grid --steps 2000000 --device cuda --out fase5_dqn_cnn

Opções principais:
    --obs     features (11 números), features_space (18 números) ou grid (tabuleiro -> CNN)
    --steps   quantos PASSOS de jogo no treino (não partidas: partidas boas são mais longas)
    --device  cpu ou cuda (placa de vídeo). Redes pequenas costumam ser mais rápidas na CPU.

Cada experimento vira uma pasta em experiments/<--out>/ (regra 4 do CLAUDE.md):
    config.json     todos os parâmetros + seed
    metrics.csv     curva de aprendizado do treino (a cada --log-every passos)
    milestones.csv  avaliação rápida (100 sementes oficiais) a cada --eval-every passos
    summary.json    avaliação final nas 1.000 sementes oficiais
    best_replay.json  a melhor partida da avaliação final
    model.pt        a rede final (pequena; o .pt não vai para o Git)
    checkpoints/    a rede em marcos do treino (0, 10 mil, 100 mil, 1 milhão...), para o vídeo
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
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.dqn import DQNAgent, ReplayBuffer  # noqa: E402
from snake import replay as rp  # noqa: E402
from snake.boards import BOARDS  # noqa: E402
from snake.evaluation import eval_seeds, evaluate, summarize  # noqa: E402
from snake.gym_env import OBSERVERS, Rewards, SnakeEnv  # noqa: E402


def epsilon_at(step: int, args) -> float:
    """Exploração cai em linha reta de eps_start até eps_end e depois fica parada."""
    progress = min(1.0, step / max(1, args.eps_decay * args.steps))
    return args.eps_start + progress * (args.eps_end - args.eps_start)


def checkpoint_marks(total: int) -> set[int]:
    """0, 10 mil, 100 mil, 1 milhão... e o fim: os momentos do treino que vão para o vídeo."""
    return {0, total} | {10**k for k in range(4, 9) if 10**k < total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", choices=list(BOARDS), default="full")
    parser.add_argument("--obs", choices=list(OBSERVERS), default="features_space")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--buffer", type=int, default=100_000, help="tamanho do replay buffer")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=10_000, help="passos aleatórios antes de treinar")
    parser.add_argument("--train-every", type=int, default=4, help="1 passo de gradiente a cada N passos de jogo")
    parser.add_argument("--target-every", type=int, default=2_000, help="copia a rede-alvo a cada N passos")
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-decay", type=float, default=0.3, help="fração do treino em que epsilon cai")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--out", required=True, help="nome da pasta em experiments/")
    args = parser.parse_args()

    out = ROOT / "experiments" / args.out
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = BOARDS[args.board]
    rewards = Rewards()
    torch.set_num_threads(1)  # rede pequena: 1 núcleo basta e deixa os outros livres
    (out / "config.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "rewards": asdict(rewards),
        "game_config": asdict(config),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    env = SnakeEnv(config, obs_type=args.obs, rewards=rewards)
    agent = DQNAgent(args.obs, config.height, config.width, gamma=args.gamma, lr=args.lr,
                     device=args.device, seed=args.seed)
    space = env.observation_space
    buffer = ReplayBuffer(args.buffer, space.shape, space.dtype, seed=args.seed)
    seed_rng = np.random.default_rng(args.seed)  # sementes de TREINO (nunca as de avaliação)
    marks = checkpoint_marks(args.steps)

    metrics_file = (out / "metrics.csv").open("w", newline="", encoding="utf-8")
    metrics = csv.writer(metrics_file)
    metrics.writerow(["step", "episodes", "epsilon", "fruits_mean", "steps_mean", "loss_mean", "seconds"])
    miles_file = (out / "milestones.csv").open("w", newline="", encoding="utf-8")
    miles = csv.writer(miles_file)
    miles.writerow(["step", "eval_fruits_mean", "eval_fruits_best", "eval_win_rate", "seconds"])
    start = time.perf_counter()

    def quick_eval(step: int) -> None:
        s = summarize(evaluate(agent, eval_seeds(100), config))
        miles.writerow([step, s["fruits_mean"], s["fruits_best"], s["win_rate"], round(time.perf_counter() - start)])
        miles_file.flush()
        print(f"  passo {step:>9,}: avaliação(100) {s['fruits_mean']:.2f} frutas (melhor {s['fruits_best']}) "
              f"| {time.perf_counter() - start:.0f} s", flush=True)

    print(f"Treinando DQN {args.obs}/{args.board} por {args.steps:,} passos ({args.device}) -> experiments/{args.out}")
    agent.save(out / "checkpoints" / "dqn_step0.pt")
    quick_eval(0)
    obs, _ = env.reset(seed=int(seed_rng.integers(0, 2**31 - 1)))
    episodes = 0
    fruits, lengths, losses = [], [], []
    for step in range(1, args.steps + 1):
        epsilon = epsilon_at(step, args)
        action = agent.choose(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # Truncada (tempo) NÃO é fim "de verdade": o futuro continua valendo (fase-4.md).
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            episodes += 1
            fruits.append(info["fruits"])
            lengths.append(info["steps"])
            obs, _ = env.reset(seed=int(seed_rng.integers(0, 2**31 - 1)))

        if step >= args.learning_starts and step % args.train_every == 0:
            losses.append(agent.train_step(buffer.sample(args.batch)))
        if step % args.target_every == 0:
            agent.sync_target()

        if step % args.log_every == 0:
            metrics.writerow([step, episodes, round(epsilon, 4),
                              round(float(np.mean(fruits)), 3) if fruits else "",
                              round(float(np.mean(lengths)), 1) if lengths else "",
                              round(float(np.mean(losses)), 5) if losses else "",
                              round(time.perf_counter() - start, 1)])
            metrics_file.flush()
            fruits, lengths, losses = [], [], []
        if step in marks:
            agent.save(out / "checkpoints" / f"dqn_step{step}.pt")
        if step % args.eval_every == 0 or step == args.steps:
            quick_eval(step)
    metrics_file.close()
    miles_file.close()
    train_seconds = time.perf_counter() - start
    agent.save(out / "model.pt")

    print(f"Treino: {train_seconds / 60:.1f} min. Avaliando nas {args.eval_games} sementes oficiais...")
    results = evaluate(agent, eval_seeds(args.eval_games), config)
    summary = summarize(results)
    summary["train_seconds"] = round(train_seconds, 1)
    summary["train_episodes"] = episodes
    summary["parameters"] = sum(p.numel() for p in agent.q_net.parameters())
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best = max(results, key=lambda r: (r["fruits"], -r["steps"]))
    replay = rp.Replay(seed=best["seed"], actions=best["actions"], config=asdict(config), player=f"dqn_{args.obs}")
    replay.result = rp.result_of(rp.simulate(replay))
    rp.save(replay, out / "best_replay.json")

    s = summary
    print(f"Avaliação final: média {s['fruits_mean']:.2f} ± {2 * s['fruits_sem']:.2f} | mediana {s['fruits_median']:g} "
          f"| melhor {s['fruits_best']} | vitórias {100 * s['win_rate']:.1f}% | {s['parameters']:,} pesos na rede")
    print("Fim: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in s["end_reasons"].items()))


if __name__ == "__main__":
    main()
