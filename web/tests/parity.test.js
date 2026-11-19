// Teste de paridade: o motor e a observação em JS respondem igual ao Python?
// Rodar com:  node --test web/tests/parity.test.js
// (node:test é o executor de testes que já vem no Node, parecido com o Jest.)
//
// Os gabaritos (fixtures.json) vêm de scripts/make_web_fixtures.py.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { SnakeGame, makeRng } from "../snake.js";
import { observe } from "../features.js";

const { cases } = JSON.parse(readFileSync(new URL("./fixtures.json", import.meta.url), "utf8"));

function gameFrom(state) {
  const game = new SnakeGame(0);
  game.loadState(state);
  return game;
}

test("há gabaritos suficientes", () => {
  assert.ok(cases.length >= 100, `só ${cases.length} estados`);
});

test("os 18 números da observação batem com o Python", () => {
  for (const [i, c] of cases.entries()) {
    const got = observe(gameFrom(c.state));
    assert.equal(got.length, 18);
    for (let k = 0; k < 18; k++) {
      // Float32 dos dois lados: diferença só de arredondamento.
      assert.ok(Math.abs(got[k] - c.features[k]) < 1e-6, `caso ${i}, número ${k}: ${got[k]} != ${c.features[k]}`);
    }
  }
});

test("direções seguras batem", () => {
  for (const c of cases) assert.deepEqual(gameFrom(c.state).safeDirections(), c.safe);
});

test("células válidas para obstáculo batem (regras 4.1 a 4.5)", () => {
  for (const c of cases) assert.deepEqual(gameFrom(c.state).eligibleObstacleCells(), c.eligibleObstacles);
});

test("um passo em cada direção dá o mesmo resultado", () => {
  for (const c of cases) {
    for (let action = 0; action < 4; action++) {
      const game = gameFrom(c.state);
      const result = game.step(action);
      const want = c.steps[action];
      assert.equal(result.endReason ?? null, want.endReason);
      assert.equal(result.ateFruit, want.ateFruit);
      assert.deepEqual(game.snake, want.snake);
    }
  }
});

test("1.000 partidas aleatórias no JS respeitam as invariantes", () => {
  for (let seed = 0; seed < 1000; seed++) {
    const game = new SnakeGame(seed);
    const pick = makeRng(seed + 99_999);
    while (!game.gameOver) {
      const safe = game.safeDirections();
      game.step(safe.length ? safe[pick(safe.length)] : game.direction);
      const all = [...game.snake, ...game.fruits, ...game.obstacles].map(String);
      assert.equal(new Set(all).size, all.length, "células sobrepostas");
      assert.ok(game.fruits.length === 5 || game.emptyCells().length === 0 || game.gameOver);
      assert.equal(game.endReason === "win", game.gameOver && game.isWon());
    }
  }
});
