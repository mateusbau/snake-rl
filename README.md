# Snake RL

Agente de aprendizado por reforço que aprende a jogar uma versão do Snake com
obstáculos permanentes (inspirada no Snake do Google). Projeto de portfólio.

> Em construção — o progresso de cada fase fica em `docs/fases/`.

**Referência humana (Fase 2):** em 20 partidas, eu fiz **16,4 frutas de média**, com
**recorde de 34**. É isso que a IA vai tentar bater.

![Minha partida recorde](docs/media/fase2_recorde_humano.gif)

**Agentes sem ML (Fase 3), 1.000 partidas cada nas mesmas sementes:**

| Jogador | Média de frutas | Melhor | Vitórias |
|---|---:|---:|---:|
| Aleatório | 0,94 | 6 | 0% |
| Aleatório seguro | 7,53 | 21 | 0% |
| Eu (humano, 20 partidas) | 16,40 | 34 | 0% |
| Guloso | 18,18 | 40 | 0% |
| BFS + flood fill | **50,78** | 69 | **1,8%** |

![Comparação dos agentes](docs/media/fase3_comparacao.png)

**Primeiro RL (Fase 4): Q-learning com tabela.** Com o tabuleiro inteiro como estado,
ele aprende no 5×5, mas no jogo oficial a tabela explode: 836 mil estados, e mesmo
assim 56% das situações da avaliação são novas. Com um resumo de 11 números, ele
chega a 17,4 frutas de média. O próximo passo é uma rede neural (DQN).

![Q-learning: aprendizado e tamanho da tabela](docs/media/fase4_qlearning.png)

## Como rodar (Windows / PowerShell)

```powershell
# 1. Criar o ambiente virtual com Python 3.12
py -3.12 -m venv .venv

# 2. Instalar as dependências
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Rodar os testes
.venv\Scripts\python.exe -m pytest

# 4. Jogar! (setas ou WASD; Espaço pausa; Esc sai)
.venv\Scripts\python.exe scripts\play.py

# 5. Ver as estatísticas das suas partidas gravadas (e gerar o resumo das 20 oficiais)
.venv\Scripts\python.exe scripts\human_stats.py

# 6. Estressar o motor do jogo com 10.000 partidas aleatórias
.venv\Scripts\python.exe scripts\random_games.py --games 10000

# 7. Avaliar os agentes nas 1.000 sementes oficiais (tabela + gráfico, ~80 s)
.venv\Scripts\python.exe scripts\evaluate.py

# 8. Treinar um Q-learning com tabela (Fase 4) no tabuleiro 5×5 (~10 s)
.venv\Scripts\python.exe scripts\train_qlearning.py --board mini --obs features --episodes 20000 --out teste_q
```

## Usando o motor do jogo

```python
from snake.env import SnakeGame, UP, RIGHT, DOWN, LEFT

game = SnakeGame(seed=42)        # mesma seed = mesma partida
result = game.step(RIGHT)        # anda uma casa
print(game)                      # H cabeça, o corpo, * fruta, # obstáculo
print(game.get_state())          # estado completo (seção 8 da especificação)
```

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `snake/` | Motor do jogo e interface com o Gymnasium |
| `agents/` | Agentes: aleatório, guloso, BFS (prontos); Q-learning, DQN, PPO (próximas fases) |
| `scripts/` | Jogar, treinar, avaliar, gerar vídeos |
| `tests/` | Testes automatizados (pytest) |
| `replays/` | Partidas gravadas (seed + ações), reproduzíveis |
| `docs/` | Especificação, relatório de cada fase e diário |
| `experiments/` | Um diretório por experimento de treino |
