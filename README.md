# Snake RL

Agente de aprendizado por reforço que aprende a jogar uma versão do Snake com
obstáculos permanentes (inspirada no Snake do Google). Projeto de portfólio.

> Em construção — o progresso de cada fase fica em `docs/fases/`.

## Como rodar (Windows / PowerShell)

```powershell
# 1. Criar o ambiente virtual com Python 3.12
py -3.12 -m venv .venv

# 2. Instalar as dependências
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Rodar os testes
.venv\Scripts\python.exe -m pytest
```

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `snake/` | Motor do jogo e interface com o Gymnasium |
| `agents/` | Agentes: aleatório, guloso, BFS, Q-learning, DQN, PPO |
| `scripts/` | Jogar, treinar, avaliar, gerar vídeos |
| `tests/` | Testes automatizados (pytest) |
| `docs/` | Especificação, relatório de cada fase e diário |
| `experiments/` | Um diretório por experimento de treino |
