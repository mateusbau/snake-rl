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
