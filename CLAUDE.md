# CLAUDE.md — Projeto Snake RL

## Quem sou e como quero trabalhar

Sou o Mateus, estudante de Ciência de Dados e IA (UFSCar). Sei JavaScript, HTML e CSS, mas sou **iniciante em Python e em machine learning**. Este é um projeto de **portfólio** para o LinkedIn e o GitHub.

- **Você (Claude Code) escreve e executa o código.** Meu papel é rodar, entender cada processo e analisar o desempenho da cobra.
- **Tudo precisa ser didático.** Comente o código em português, explicando o porquê, não só o quê. Quando um conceito de Python tiver equivalente em JavaScript, compare.
- **Ao fim de cada fase**, escreva `docs/fases/fase-N.md` em português com: o que foi feito; os conceitos novos, explicados com exemplos práticos e pouca teoria (aprendo melhor com exemplos resolvidos passo a passo); como rodar; os resultados (números e gráficos); o que deu errado e como foi resolvido.
- **Pare ao fim de cada fase** e me mostre um resumo antes de começar a próxima.
- **Ambiente:** Windows 11, PowerShell, Python 3.12.10 em `.venv`. Há também um Python 3.14 instalado no sistema, que **não** deve ser usado. Sempre use o Python da `.venv`.
- **Idioma:** nomes no código em inglês; comentários, documentação e commits em português.

## O jogo

A especificação completa está em `docs/especificacao.md`. **Leia antes de qualquer coisa e respeite integralmente.** Resumo: tabuleiro 9×10, cobra inicial de 3 células, sempre 5 frutas, e 1 obstáculo permanente gerado a cada quantidade ímpar de frutas coletadas. Um obstáculo não pode nascer sobre a cobra, uma fruta ou outro obstáculo; não pode encostar em outro obstáculo nas 8 direções; e não pode ficar nas 3 células à frente da cabeça. A posição é sorteada de forma uniforme entre as células válidas. Colisão com a parede, o corpo ou um obstáculo = derrota.

### Decisões que completam a especificação (já confirmadas)

| Ponto | Decisão |
|---|---|
| Vitória ("zerar") | Não existe célula livre para repor fruta |
| Crescimento | +1 célula por fruta |
| Comando de 180° | Ignorado; mantém a direção atual |
| Limite de passos sem comer | Encerra a partida (configurável; padrão 200) |
| Frutas iniciais | **Sempre fixas** em (linha, coluna): (2,4), (2,8), (4,6), (6,4), (6,8). Só as reposições são sorteadas |
| Posição inicial | Linha 4, corpo nas colunas 1–2, cabeça na coluna 3, direção direita (índices a partir de 0). Deixar configurável |
| Ordem dentro de um passo | Mover → checar colisão → comer → repor fruta → gerar obstáculo (se ímpar) |
| Modos | Sem tela (treino rápido) e com tela em pygame (eu jogo e assisto ao agente) |
| Aleatoriedade | Sempre por semente (seed), totalmente reproduzível |
| Entrar na casa da cauda | Permitido se a cobra não cresce no passo (a cauda sai junto) |
| Becos sem saída (seção 4.5) | Proibidos: um obstáculo só nasce se toda célula livre continuar com ≥ 2 vizinhas ortogonais livres (só paredes e obstáculos bloqueiam). **Decisão nossa; diverge do Snake do Google.** Replays gravados antes dela são reproduzidos com `forbid_dead_ends=False` |

Observações úteis: os obstáculos nunca passam de 25 (não podem se encostar); na prática o mapa satura com 13–20 (média 16,5). Sozinhos, eles nunca isolam regiões do mapa nem criam becos; o que prende a cobra é o corpo somado aos obstáculos.

## Objetivo e métricas

- **M1:** a média de frutas do agente supera a minha média humana.
- **M2:** o agente supera o meu recorde (hoje 22 frutas) com frequência.
- **M3:** zerar o jogo, medido como taxa de vitória.

Métricas de toda avaliação: média e mediana de frutas, melhor partida, % do tabuleiro preenchido, causa da morte e taxa de vitória. **Toda avaliação usa o mesmo conjunto fixo de 1.000 sementes**, via `scripts/evaluate.py`.

## Fases

| Fase | Entrega | Concluída quando |
|---|---|---|
| 0. Preparação | Estrutura, `.venv`, Git, GitHub | `pytest` passa e o repositório está no GitHub |
| 1. Motor do jogo | `snake/env.py` sem tela e sem ML | Os 15 testes da seção 10 da especificação passam e 10.000 partidas aleatórias rodam sem erro |
| 2. Versão jogável | `snake/render.py` (pygame), `scripts/play.py`, gravação de replays | **Pausar:** eu jogo 20 partidas e registramos minha média e meu recorde |
| 3. Baselines sem ML | Agentes aleatório, guloso e BFS + flood fill; `scripts/evaluate.py` | Tabela comparando os agentes e eu |
| 4. RL em miniatura | Interface Gymnasium (`snake/gym_env.py`); Q-learning com tabela num 5×5 sem obstáculos | O agente aprende, e a doc explica por que a tabela não escala |
| 5. DQN do zero | PyTorch: vetor de características, depois CNN com o tabuleiro em canais | O DQN supera o agente guloso |
| 6. PPO | Stable-Baselines3 / MaskablePPO, ambientes em paralelo, currículo | M1 e M2 atingidos |
| 7. Rumo a zerar | Rede + busca (checagem de segurança, lookahead, talvez MCTS) | Taxa de vitória medida e limites explicados |
| 8. Portfólio | README final, gráficos, vídeo, demo web (ONNX + JS) | Links prontos para o LinkedIn |

Antes de treinar (Fase 4 em diante), verifique o hardware (CPU/GPU) e mantenha os tempos de treino razoáveis para um notebook. Avise antes de qualquer treino que vá passar de 30 minutos.

## Estrutura de pastas

```
snake-rl/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── docs/            # especificacao.md, fases/, diario.md
├── snake/           # env.py, gym_env.py, render.py
├── agents/          # random, greedy, bfs, qlearning, dqn, ppo
├── scripts/         # play.py, train.py, evaluate.py, make_video.py
├── tests/
├── experiments/     # uma pasta por experimento: config, seed, métricas, checkpoints
└── web/             # demo no navegador (Fase 8)
```

## Regras de trabalho

1. Uma fase por vez, com o critério de conclusão atingido antes de avançar.
2. **O ambiente é congelado depois da Fase 1.** Se uma regra mudar, os testes mudam primeiro.
3. Todo número vem do `evaluate.py` com as mesmas sementes. Nada de "parece que melhorou".
4. Um experimento = uma mudança. Cada treino salva config, seed e resultados em `experiments/`.
5. `docs/diario.md`: registro curto de cada decisão, tentativa e erro. Vai virar o texto do portfólio.
6. Commit ao fim de cada etapa relevante, com mensagem em português dizendo o que mudou.
7. Mantenha `requirements.txt` atualizado e o README com "como rodar".

## Material visual para o LinkedIn (preparar desde o início)

O objetivo final é um vídeo de 45–75 s, quadrado ou vertical, com legendas: gancho com a IA quase zerando → obstáculos surgindo → **tela dividida em 4 com a mesma semente em 4 momentos do treino** → curva de aprendizado animada → gráfico de barras (aleatório, guloso, BFS, eu, DQN, PPO) → taxa de vitória e links. Para isso:

- **Checkpoints** do modelo em marcos fixos do treino (ex.: 0, 10 mil, 100 mil, 1 milhão de passos…).
- **Replays** salvos como semente + lista de ações, renderizáveis depois em alta qualidade.
- **Métricas em CSV** durante o treino, para gerar gráficos e animações.
- **Minhas partidas gravadas** na Fase 2, para a comparação humano × IA.
- `scripts/make_video.py`: gera os quadros (pygame + matplotlib) e junta num MP4 com ffmpeg.
- Visual do jogo inspirado no Snake do Google (tabuleiro quadriculado verde, cobra azul, frutas vermelhas, obstáculos verde-escuro), mas com identidade própria.
- Cada fase gera pelo menos um GIF ou gráfico em `docs/media/`, para posts menores ao longo do projeto.
