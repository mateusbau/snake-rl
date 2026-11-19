"""Exporta a rede do PPO para ONNX, o formato que o navegador sabe rodar (Fase 8).

O que é ONNX? Um formato de arquivo "neutro" para redes neurais, como o PDF é para
documentos: o PyTorch salva, e outros programas (aqui, o onnxruntime-web, em
JavaScript) abrem sem precisar do Python. Assim a demo roda 100% no navegador.

Só exportamos o ATOR (a parte da rede que escolhe a ação). O crítico (que estima o
valor do estado) só serve durante o treino.

    18 números (observação) -> Linear 128 -> tanh -> Linear 128 -> tanh -> Linear 4 -> "logits"

"Logits" são as notas cruas das 4 ações. O PPO joga com a de maior nota
(deterministic=True), e o JS faz o mesmo: argmax. Para mostrar probabilidades
na tela, o JS aplica softmax (exp de cada nota, dividido pela soma).

Como rodar:
    .venv\\Scripts\\python.exe scripts\\export_onnx.py
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agents.ppo import PPOAgent  # noqa: E402

# Modelo final + os marcos do treino (a demo deixa trocar entre eles).
MODELS = {
    "final": "experiments/fase6_ppo/model.zip",
    "1m": "experiments/fase6_ppo/checkpoints/ppo_step1000000.zip",
    "100k": "experiments/fase6_ppo/checkpoints/ppo_step100000.zip",
    "10k": "experiments/fase6_ppo/checkpoints/ppo_step10000.zip",
    "0": "experiments/fase6_ppo/checkpoints/ppo_step0.zip",
}


class Actor(torch.nn.Module):
    """Só o caminho observação -> logits, copiado do modelo do SB3.

    Herdar de torch.nn.Module é como "extends" numa classe JS: ganhamos tudo que
    o PyTorch precisa para exportar, e só escrevemos o forward (a conta da rede).
    """

    def __init__(self, policy):
        super().__init__()
        self.body = policy.mlp_extractor.policy_net   # Linear-tanh-Linear-tanh
        self.head = policy.action_net                 # Linear 128 -> 4

    def forward(self, obs):
        return self.head(self.body(obs))


def export(zip_path: Path, out: Path) -> None:
    agent = PPOAgent.load(zip_path)
    if agent.obs_type != "features_space" or agent.masked:
        raise SystemExit("A demo web espera um PPO sem máscara com observação features_space (18 números).")
    actor = Actor(agent.model.policy).eval()
    example = torch.zeros(1, 18)
    out.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False usa o exportador clássico (mais simples e estável para uma MLP).
    # dynamic_axes: o tamanho do lote (1ª dimensão) pode variar; o JS manda 1 por vez.
    torch.onnx.export(actor, example, out, input_names=["obs"], output_names=["logits"],
                      dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    check(actor, out)


def check(actor: torch.nn.Module, onnx_path: Path) -> None:
    """Confere que o ONNX dá as MESMAS notas que o PyTorch em 1.000 observações aleatórias."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    obs = np.random.default_rng(0).random((1000, 18), dtype=np.float32)
    with torch.no_grad():
        expected = actor(torch.from_numpy(obs)).numpy()
    got = session.run(["logits"], {"obs": obs})[0]
    diff = float(np.abs(expected - got).max())
    same_choice = float((expected.argmax(1) == got.argmax(1)).mean())
    assert diff < 1e-4 and same_choice == 1.0, f"ONNX diverge do PyTorch: {diff=}, {same_choice=}"
    print(f"{onnx_path.name}: {onnx_path.stat().st_size / 1024:.0f} KB, diferença máxima {diff:.1e}, mesma escolha em 100%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "models")
    args = parser.parse_args()
    for name, path in MODELS.items():
        export(ROOT / path, args.out / f"ppo_{name}.onnx")
    (args.out / "models.json").write_text(json.dumps(list(MODELS), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
