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
