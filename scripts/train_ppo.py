"""Treina um PPO / MaskablePPO (Fase 6) com vários tabuleiros em paralelo.

Como rodar (na raiz do projeto):
    .venv\\Scripts\\python.exe scripts\\train_ppo.py --steps 5000000 --out fase6_ppo
    .venv\\Scripts\\python.exe scripts\\train_ppo.py --steps 5000000 --masked --out fase6_maskppo

Opções principais:
    --obs      features_space (18 números, padrão), features ou grid
    --masked   usa o MaskablePPO (proíbe as jogadas que matam na hora)
    --n-envs   quantos tabuleiros jogando ao mesmo tempo (um processo cada)
    --steps    total de passos de jogo, somando todos os tabuleiros

Saídas em experiments/<--out>/ (regra 4): config.json, metrics.csv, milestones.csv,
summary.json, best_replay.json, model.zip + model.json, checkpoints/ (marcos para o vídeo).
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

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from agents.ppo import NET_ARCH, PPO_DEFAULTS, EnvFactory, PPOAgent, build_model  # noqa: E402
from snake import replay as rp  # noqa: E402
from snake.boards import BOARDS  # noqa: E402
from snake.curriculum import collect_start_states  # noqa: E402
from snake.evaluation import eval_seeds, evaluate, summarize  # noqa: E402
from snake.gym_env import OBSERVERS, Rewards  # noqa: E402


class ProgressCallback(BaseCallback):
    """Chamado pelo SB3 a cada passo (de todos os ambientes juntos).

    Junta as partidas que terminam (para a curva de treino), avalia nas 100 primeiras
    sementes oficiais a cada `eval_every` passos e salva checkpoints nos marcos.
    É como um "event listener" que o framework chama durante o laço de treino.
    """

    def __init__(self, args, out: Path, config):
        super().__init__()
        self.args, self.out, self.config = args, out, config
        self.marks = sorted({10**k for k in range(4, 9) if 10**k < args.steps})
        self.next_log, self.next_eval = args.log_every, args.eval_every
        self.fruits, self.lengths = [], []
        self.episodes = 0
        self.start = time.perf_counter()
        self.metrics_file = (out / "metrics.csv").open("w", newline="", encoding="utf-8")
        self.metrics = csv.writer(self.metrics_file)
        self.metrics.writerow(["step", "episodes", "fruits_mean", "steps_mean", "entropy", "seconds"])
        self.miles_file = (out / "milestones.csv").open("w", newline="", encoding="utf-8")
        self.miles = csv.writer(self.miles_file)
        self.miles.writerow(["step", "eval_fruits_mean", "eval_fruits_best", "eval_win_rate", "seconds"])

    def agent(self) -> PPOAgent:
        return PPOAgent(self.model, self.args.obs, self.args.masked, self.args.board)

    def quick_eval(self) -> None:
        s = summarize(evaluate(self.agent(), eval_seeds(100), self.config))
        elapsed = time.perf_counter() - self.start
        self.miles.writerow([self.num_timesteps, s["fruits_mean"], s["fruits_best"], s["win_rate"], round(elapsed)])
        self.miles_file.flush()
        print(f"  passo {self.num_timesteps:>10,}: avaliação(100) {s['fruits_mean']:.2f} frutas "
              f"(melhor {s['fruits_best']}, vitórias {s['win_rate']:.0%}) | {elapsed / 60:.1f} min", flush=True)

    def _on_training_start(self) -> None:
        self.agent().save(self.out / "checkpoints" / "ppo_step0.zip")
        self.quick_eval()

    def _on_step(self) -> bool:
        # self.locals = variáveis do laço do SB3 neste passo (um dicionário).
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episodes += 1
                self.fruits.append(info["fruits"])
                self.lengths.append(info["steps"])
        if self.marks and self.num_timesteps >= self.marks[0]:
            self.agent().save(self.out / "checkpoints" / f"ppo_step{self.marks.pop(0)}.zip")
        if self.num_timesteps >= self.next_log:
            self.next_log += self.args.log_every
            entropy = self.logger.name_to_value.get("train/entropy_loss", float("nan"))
            self.metrics.writerow([self.num_timesteps, self.episodes,
                                   round(float(np.mean(self.fruits)), 3) if self.fruits else "",
                                   round(float(np.mean(self.lengths)), 1) if self.lengths else "",
                                   round(-entropy, 4), round(time.perf_counter() - self.start, 1)])
            self.metrics_file.flush()
            self.fruits, self.lengths = [], []
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.args.eval_every
            self.quick_eval()
        return True  # False interromperia o treino

    def _on_training_end(self) -> None:
        self.metrics_file.close()
        self.miles_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", choices=list(BOARDS), default="full")
    parser.add_argument("--obs", choices=[o for o in OBSERVERS if o != "grid"], default="features_space")
    parser.add_argument("--masked", action="store_true")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=PPO_DEFAULTS["learning_rate"])
    parser.add_argument("--gamma", type=float, default=PPO_DEFAULTS["gamma"])
    parser.add_argument("--ent-coef", type=float, default=PPO_DEFAULTS["ent_coef"])
    parser.add_argument("--curriculum", action="store_true",
                        help="parte das partidas de treino começa avançada (snake/curriculum.py)")
    parser.add_argument("--start-prob", type=float, default=0.5, help="currículo: chance de começar avançada")
    parser.add_argument("--curriculum-games", type=int, default=300, help="currículo: partidas do BFS no banco")
    parser.add_argument("--min-fruits", type=int, default=20, help="currículo: só situações com >= N frutas")
    parser.add_argument("--timeout-reward", type=float, default=Rewards().timeout,
                        help="recompensa quando o tempo sem comer esgota (padrão 0; Fase 7 testa -1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=500_000)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = ROOT / "experiments" / args.out
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = BOARDS[args.board]
    torch.set_num_threads(1)
    rewards = Rewards(timeout=args.timeout_reward)
    overrides = {"learning_rate": args.lr, "gamma": args.gamma, "ent_coef": args.ent_coef}
    start_states, curriculum_info = None, None
    if args.curriculum:
        t0 = time.perf_counter()
        start_states, cur_seeds = collect_start_states(args.curriculum_games, seed=args.seed + 1000,
                                                       min_fruits=args.min_fruits, board=args.board)
        fruits = [s["fruits_collected"] for s in start_states]
        curriculum_info = {"states": len(start_states), "bfs_games": len(cur_seeds),
                           "fruits_min": min(fruits), "fruits_mean": round(float(np.mean(fruits)), 1),
                           "fruits_max": max(fruits), "seconds": round(time.perf_counter() - t0, 1)}
        print(f"Currículo: {curriculum_info}", flush=True)
    (out / "config.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "curriculum": curriculum_info,
        "ppo": {**PPO_DEFAULTS, **overrides, "net_arch": NET_ARCH},
        "rewards": asdict(rewards),
        "game_config": asdict(config),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Cada tabuleiro roda num processo separado (usa vários núcleos da CPU).
    prob = args.start_prob if args.curriculum else 0.0
    env = SubprocVecEnv([EnvFactory(args.board, args.obs, start_states, prob, rewards) for _ in range(args.n_envs)])
    env.seed(args.seed)  # sementes de TREINO: seed, seed+1, ... (sorteiam as partidas)
    model = build_model(env, args.masked, args.seed, **overrides)
    algo = "MaskablePPO" if args.masked else "PPO"
    print(f"Treinando {algo} {args.obs}/{args.board} por {args.steps:,} passos "
          f"({args.n_envs} tabuleiros em paralelo) -> experiments/{args.out}", flush=True)
    callback = ProgressCallback(args, out, config)
    model.learn(total_timesteps=args.steps, callback=callback)
    env.close()
    train_seconds = time.perf_counter() - callback.start

    agent = PPOAgent(model, args.obs, args.masked, args.board)
    agent.save(out / "model.zip")
    print(f"Treino: {train_seconds / 60:.1f} min. Avaliando nas {args.eval_games} sementes oficiais...", flush=True)
    results = evaluate(agent, eval_seeds(args.eval_games), config)
    summary = summarize(results)
    summary["train_seconds"] = round(train_seconds, 1)
    summary["train_episodes"] = callback.episodes
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best = max(results, key=lambda r: (r["fruits"], -r["steps"]))
    replay = rp.Replay(seed=best["seed"], actions=best["actions"], config=asdict(config), player=f"ppo_{args.out}")
    replay.result = rp.result_of(rp.simulate(replay))
    rp.save(replay, out / "best_replay.json")

    s = summary
    print(f"Avaliação final: média {s['fruits_mean']:.2f} ± {2 * s['fruits_sem']:.2f} | mediana {s['fruits_median']:g} "
          f"| melhor {s['fruits_best']} | vitórias {100 * s['win_rate']:.1f}%")
    print("Fim: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in s["end_reasons"].items()))


# No Windows, processos novos reimportam este arquivo: sem este "if", cada um
# tentaria começar outro treino (e criar mais processos, sem fim).
if __name__ == "__main__":
    main()
