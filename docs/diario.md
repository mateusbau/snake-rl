# Diário do projeto

Registro curto de decisões, tentativas e erros.

## 2026-11-17 — Fase 0

- Primeiro commit tinha incluído a `.venv` inteira (~2.000 arquivos) porque o
  `.gitignore` estava vazio. Desfeito antes de publicar e refeito com o
  `.gitignore` correto.
- As pastas `agents/`, `docs/` e `tests/` tinham sido criadas aninhadas dentro de
  `snake/` por engano; movidas para a raiz, como no CLAUDE.md.

## 2026-11-17 — Fase 1

- Regra nova confirmada: entrar na casa da cauda é permitido quando a cobra não
  cresce no passo (a cauda sai junto). Não estava na especificação nem no CLAUDE.md.
- Motor com 22 testes passando; 20.000 partidas aleatórias sem invariante quebrada.
- Saturação: em média só cabem 19 obstáculos (não 25) quando sorteados aleatoriamente.
- Tamanho do crescimento por fruta (`growth_per_fruit`) ficou configurável, como pede
  a seção 6.2 da especificação; a frequência de obstáculos NÃO é configurável (seção 11).

## 2026-11-17 — Fase 2

- Versão jogável em pygame, com fila de comandos para curvas rápidas.
- Replays como seed + ações (poucos KB), conferidos por re-simulação.
- Velocidade padrão 7 passos/s (ajustável com --speed); a cobra espera o 1º toque.
- Partidas humanas ficam em `replays/human/` e entram no Git (são dados do projeto).

## 2026-11-17 — Nova regra: proibição de becos sem saída (seção 4.5)

- **Problema (achado jogando):** um obstáculo podia criar uma célula livre com uma
  única saída (ex.: encostado num canto). Sem meia-volta, entrar nela é morte certa.
- **Medição com a regra antiga** (10.000 partidas `safe_random`): 36,7% das partidas
  tiveram pelo menos um beco; 24,5% dos obstáculos nasceram criando um.
- **Regra nova:** um obstáculo só nasce se toda célula livre continuar com ≥ 2
  vizinhas ortogonais livres. Só paredes e obstáculos bloqueiam (cobra e frutas
  contam como livres). **Decisão nossa; diverge do Snake do Google.**
- Processo: testes primeiro (16 a 20 na seção 10; 5 falharam), depois a implementação.
- **Efeito nos obstáculos:** nas partidas aleatórias, média e máximo por partida
  quase não mudaram (`safe_random`: 3,81 → 3,96 de média, máximo 12 → 12), porque
  essas partidas nunca chegam perto de lotar o mapa e a regra muda *onde* o
  obstáculo nasce, não *quantos*. A subida da média vem de o jogador comer mais
  (7,15 → 7,46 frutas). A saturação mudou: o mapa agora lota com 16,5 obstáculos
  em média (13–20), contra 19,0 (15–23) antes.
- Mortes na parede do `safe_random` caíram de 42,2% para 35,1%: os becos de canto sumiram.
- Desempenho: a regra deixou o motor ~28% mais lento (68 mil → 49 mil passos/s),
  depois de duas otimizações (conjunto em vez de lista; contar as saídas uma vez).
- As 24 partidas humanas já gravadas foram jogadas com a regra antiga. Elas
  continuam reproduzíveis (`forbid_dead_ends=False` para replays sem essa chave) e
  aparecem separadas no `human_stats.py`. As 20 partidas oficiais da Fase 2 devem ser
  jogadas com a regra nova.

## 2026-11-18 — Fase 2 concluída: referência humana

- 20 partidas oficiais com a regra atual: **média 16,40, mediana 14, recorde 34**,
  desvio padrão 8,64. Mortes: obstáculo 17, parede 3. Nenhuma vitória.
- Definição: oficiais = as 20 primeiras partidas com a regra 4.5, em ordem
  cronológica (decidido antes de olhar os números, para não escolher as melhores).
  1 partida extra e as 24 da regra antiga ficam só como histórico.
- A média com a regra antiga era 10,79. Prática e regra nova mudaram juntas, então
  não dá para atribuir a diferença a uma só.
- Erro padrão da média ≈ 1,9 (intervalo de ~95%: 12,5 a 20,3). Um agente só "bate a
  M1" de forma convincente se ficar bem acima de 16,4.
- Resumo salvo em `docs/human_baseline.json`; gráfico `fase2_humano.png`; GIF do
  recorde `fase2_recorde_humano.gif`. M1 e M2 do `CLAUDE.md` atualizadas (16,4 / 34).

## 2026-11-18 — Fase 3: baselines sem ML

- 4 agentes (aleatório, aleatório seguro, guloso, BFS + flood fill) com a mesma
  interface `reset(seed)` / `act(game)`. Eles só consultam `get_state()` e
  `safe_directions()` (seção 9); um teste garante que não alteram o jogo nem o gerador.
- `snake/evaluation.py`: 1.000 sementes oficiais derivadas da semente-mestra 20261118
  (impressão digital `158f340ed1ca`). Nunca mudar.
- Resultados (1.000 partidas): aleatório 0,94 · aleatório seguro 7,53 · guloso 18,18 ·
  **BFS 50,78 (melhor 69, 1,8% de vitórias)**. Humano: 16,40 (20 partidas).
- O guloso empata com o humano (dentro da margem); o BFS já bate M1 e M2. A régua
  real para o RL passa a ser o BFS e a taxa de vitória.
- BFS termina por tempo em 83% das partidas, quase sempre no fim do jogo (54,4 frutas
  em média, ~10 células vazias): comer qualquer fruta restante o encurralaria, então
  ele segue a cauda até o limite de 200 passos.
- Tentativas para reduzir os "tempos" (100 sementes): (a) aceitar comer se o espaço
  livre ≥ tamanho da cobra: sem efeito (51,3 → 51,4); (b) teste da cauda "com tempo":
  pior (49,4 frutas, mortes de 15% → 52%). Mantida a versão original.


## 2026-11-18 — Fase 4: Gymnasium + Q-learning com tabela

- Hardware: Ryzen 5 5600X (6 núcleos), 16 GB, RTX 2060 SUPER 8 GB. A tabela usa só a CPU.
- `snake/gym_env.py`: `SnakeEnv` (Gymnasium 1.4) com observação `grid` (4 camadas 0/1)
  ou `features` (11 números), recompensas +1 fruta, −1 morte, +10 vitória, 0 por passo;
  limite de tempo = `truncated` (não é morte). Passa no `check_env` oficial.
- Mudança no motor, com testes primeiro: `GameConfig.spawn_obstacles` (padrão True),
  só para o tabuleiro de treino `mini` (5×5, 1 fruta). Conflito resolvido entre a
  seção 11 (não mudar a frequência) e o CLAUDE.md (5×5 sem obstáculos).
- 4 experimentos, mesmos parâmetros (alpha 0,1, gamma 0,9, epsilon 1 → 0,05, seed 0):
  - grid/mini, 1 milhão de partidas (316 s): 7,38 frutas, 380 mil estados, 3,8% nunca vistos.
  - features/mini, 20 mil (11 s): 11,06 frutas (guloso: 10,71), 256 estados.
  - features/full, 20 mil (38 s): 17,42 frutas (guloso 18,18; humano 16,40).
  - grid/full, 100 mil (63 s): 1,80 frutas, 836 mil estados (358 MB), 56% nunca vistos.
- Conclusão: a tabela aprende, mas não generaliza. O resumo generaliza, mas é cego
  ao corpo (59% das mortes). Próximo passo: rede neural (DQN), Fase 5.


## 2026-11-18 — Fase 5: DQN do zero (PyTorch)

- PyTorch 2.14 com CUDA 12.6 (o `pip install torch` comum é só-CPU). Medição de 30 mil
  passos: vetor/CPU 33 s, CNN/CPU 422 s, CNN/GPU 51 s. Redes pequenas na CPU (1 núcleo),
  CNN na GPU.
- `agents/dqn.py`: MLP (vetores), CNN (tabuleiro 4×9×10), replay buffer circular, rede-alvo,
  erro de Huber, Adam. Nova observação `features_space` = 11 números + espaço livre e
  alcance da cauda após cada jogada (flood fill) + tamanho.
- Mesmos parâmetros nos 3 experimentos (gamma 0,95, lr 5e-4, buffer 100 mil, lote 128,
  epsilon 1 → 0,02 em 30% do treino, seed 0); só muda a observação.
- Resultados oficiais (1.000 sementes, `experiments/fase5_comparacao`):
  - DQN 11 números (300 mil passos, 2,6 min): 18,43 = empate com guloso (18,18) e Q-tabela (17,42).
  - DQN CNN (600 mil passos, 14 min GPU): 24,38, melhor 48.
  - **DQN 18 números (1 milhão, 13 min): 40,99, melhor 67, 0,2% de vitórias (2 partidas);
    passa de 34 frutas em 747 das 1.000 partidas.** Critério da fase atingido; M1 e M2 também.
- Lição: com a mesma visão, rede = tabela. O ganho veio do que o agente enxerga.
- Em aberto: as curvas atingem o pico quando o epsilon termina de cair (44,7 em 300 mil
  passos) e depois oscilam ~40. Mantido o modelo final (escolher o melhor marco pelas
  sementes de avaliação seria vazamento).


## 2026-11-18 — Correção da regra de vitória

- Bug apontado pelo Mateus: o motor declarava vitória quando não havia célula vazia para
  repor fruta, contando fruta como célula ocupada. Todas as vitórias no tabuleiro oficial
  (18 do BFS, 2 do DQN) terminaram com 4 frutas sem comer. No 5×5 (1 fruta) não havia erro.
- Regra nova: vitória ⇔ cobra + obstáculos ocupam todas as células (`SnakeGame.is_won()`),
  checada no fim do passo. Sem célula vazia, não nasce fruta e a partida segue.
- Testes primeiro (regra 2): o teste 14(c) antigo codificava o bug e foi corrigido; 3 testes
  novos; `check_invariants` agora exige vitória ⇔ `is_won()`. 86 testes passam.
- Reavaliação nas 1.000 sementes: **BFS 50,79 (0% vitórias, melhor 71)**, **DQN 18 números
  41,00 (0%, melhor 69)**; demais agentes iguais. Nas 18 sementes das "vitórias" do BFS,
  todas viram fim por tempo, algumas com mais frutas (até 71).
- GIFs `*_vitoria.gif` trocados por `fase3_bfs_melhor.gif` e `fase5_dqn_melhor.gif`.
  DQNs não retreinados (bônus de vitória raríssimo no treino).


## 2026-11-18 — Fase 6: PPO (Stable-Baselines3)

- SB3 + sb3-contrib 2.9.2. `agents/ppo.py` (PPOAgent), `scripts/train_ppo.py` (8 tabuleiros em
  `SubprocVecEnv`, ~2.700 passos/s na CPU), observação de 18 números, rede 128×128, 4 milhões de passos.
- Máscara: `action_mask(game)` em `gym_env.py`; com a cobra presa, libera tudo (evita 0/0). Teste antes.
- Currículo (`snake/curriculum.py`): 5.162 situações de 300 partidas do BFS (sementes de treino),
  50% das partidas de treino começam nelas. Avaliação sempre do início oficial.
- Resultados (1.000 sementes, `experiments/fase6_comparacao`):
  - **PPO: 46,59, mediana 49, melhor 67, > 34 em 851 partidas. M1 e M2 batidas.** 15 min.
  - PPO + currículo: 45,84 (empate); o "degrau" veio em 2,5 M passos em vez de 3,5 M.
  - MaskablePPO: 20,26. Travou em ~19 desde 500 mil passos; entropia 0,15 → 0,05.
  - MaskablePPO com ent_coef 0,05: 31,67 e ainda subindo (entropia 0,2–0,3). Hipótese confirmada em parte.
- Todos os bons agentes param no fim de jogo dando voltas (83–86% por tempo). Com a recompensa
  atual, esperar custa 0 e arriscar custa −1. Zerar exige planejar: Fase 7.
- Um treino foi encerrado pelo sistema por falta de memória; reiniciado com autorização.
