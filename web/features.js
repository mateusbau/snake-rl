// Os 18 números que a rede enxerga: tradução de space_feature_observation
// (snake/gym_env.py) e de flood_fill / tail_reachable (agents/grid.py).
//
// Se UM número sair diferente do Python, a rede recebe uma "visão" que nunca viu
// no treino e joga pior sem dar erro nenhum. Por isso o teste de paridade compara
// os 18 valores com os do Python em centenas de estados reais.

import { DELTAS, key } from "./snake.js";

function* neighbors([r, c], height, width) {
  for (const [dr, dc] of DELTAS) {
    const nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < height && nc >= 0 && nc < width) yield [nr, nc];
  }
}

/** Quantas células livres a partir de `start` (sem contar ela): o "balde de tinta". */
export function floodFill(start, blocked, height, width) {
  const seen = new Set([key(start)]);
  const queue = [start];
  while (queue.length) {
    const cell = queue.shift();
    for (const nxt of neighbors(cell, height, width)) {
      const k = key(nxt);
      if (!seen.has(k) && !blocked.has(k)) { seen.add(k); queue.push(nxt); }
    }
  }
  return seen.size - 1;
}

/** A cabeça alcança a cauda? (copiado linha a linha do Python, inclusive o caso len = 2). */
export function tailReachable(body, obstacles, height, width) {
  if (body.length < 2) return true;
  const head = body[0], tailKey = key(body[body.length - 1]), headKey = key(head);
  const blocked = new Set([...obstacles, ...body.slice(0, -1).map(key)]);
  const seen = new Set([headKey]);
  const queue = [head];
  while (queue.length) {
    const cell = queue.shift();
    for (const nxt of neighbors(cell, height, width)) {
      const k = key(nxt);
      if (k === tailKey && (key(cell) !== headKey || body.length > 2)) return true;
      if (!seen.has(k) && !blocked.has(k)) { seen.add(k); queue.push(nxt); }
    }
  }
  return false;
}

export const FEATURE_NAMES = [
  "perigo frente", "perigo direita", "perigo esquerda",
  "indo p/ cima", "indo p/ direita", "indo p/ baixo", "indo p/ esquerda",
  "fruta acima", "fruta abaixo", "fruta à esquerda", "fruta à direita",
  "espaço frente", "espaço direita", "espaço esquerda",
  "cauda frente", "cauda direita", "cauda esquerda",
  "tamanho",
];

/** Float32Array com os 18 números, na mesma ordem do Python. */
export function observe(game) {
  const { height, width } = game.config;
  const dir = game.direction;
  const safe = new Set(game.safeDirections());
  const relative = [dir, (dir + 1) % 4, (dir + 3) % 4];   // frente, direita, esquerda
  const danger = relative.map((d) => (safe.has(d) ? 0 : 1));
  const oneHot = [0, 1, 2, 3].map((d) => (d === dir ? 1 : 0));

  const [hr, hc] = game.head;
  let fruit = [0, 0, 0, 0];
  if (game.fruits.length) {
    // Fruta mais próxima (distância de Manhattan); empate -> a primeira da lista, como o min() do Python.
    let best = game.fruits[0], bestDist = Infinity;
    for (const f of game.fruits) {
      const d = Math.abs(f[0] - hr) + Math.abs(f[1] - hc);
      if (d < bestDist) { best = f; bestDist = d; }
    }
    fruit = [best[0] < hr, best[0] > hr, best[1] < hc, best[1] > hc].map(Number);
  }

  const obstacles = new Set(game.obstacles.map(key));
  const fruits = new Set(game.fruits.map(key));
  const space = [], tail = [];
  for (const d of relative) {
    if (!safe.has(d)) { space.push(0); tail.push(0); continue; }
    const newHead = [hr + DELTAS[d][0], hc + DELTAS[d][1]];
    const newBody = [newHead, ...(fruits.has(key(newHead)) ? game.snake : game.snake.slice(0, -1))];
    const free = height * width - newBody.length - obstacles.size;
    const blocked = new Set([...obstacles, ...newBody.slice(1).map(key)]);
    const reach = floodFill(newHead, blocked, height, width);
    space.push(free > 0 ? Math.min(1, reach / free) : 0);
    tail.push(tailReachable(newBody, obstacles, height, width) ? 1 : 0);
  }
  const length = game.snake.length / (height * width);
  return Float32Array.from([...danger, ...oneHot, ...fruit, ...space, ...tail, length]);
}
