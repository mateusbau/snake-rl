// A demo: desenha o jogo num <canvas> e, a cada passo, pergunta à rede o que fazer.
//
// Fluxo de um passo da IA (o mesmo do Python, agents/ppo.py):
//   jogo -> observe() (18 números) -> rede ONNX -> 4 notas (logits) -> maior nota -> game.step()

import { SnakeGame, DELTAS, OPPOSITE, UP, RIGHT, DOWN, LEFT } from "./snake.js";
import { observe } from "./features.js";

// ---------------------------------------------------------------- desenho
// Mesmas cores do snake/render.py (pygame), para a demo e os vídeos combinarem.
const COLORS = {
  light: "#aad751", dark: "#a2d149", wall: "#4a752c", obstacle: "#4a752c", obstacleEdge: "#3a5e22",
  head: [78, 124, 246], tail: [36, 66, 158], fruit: "#e7471d", leaf: "#388e3c",
};
const CELL = 48, BORDER = 12;
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");

function cellXY([r, c]) { return [BORDER + c * CELL, BORDER + r * CELL]; }

function draw(game, overlay = null) {
  const { height, width } = game.config;
  ctx.fillStyle = COLORS.wall;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < height; r++)
    for (let c = 0; c < width; c++) {
      ctx.fillStyle = (r + c) % 2 === 0 ? COLORS.light : COLORS.dark;
      ctx.fillRect(...cellXY([r, c]), CELL, CELL);
    }
  for (const cell of game.obstacles) {
    const [x, y] = cellXY(cell);
    ctx.fillStyle = COLORS.obstacle;
    ctx.fillRect(x, y, CELL, CELL);
    ctx.strokeStyle = COLORS.obstacleEdge;
    ctx.lineWidth = 3;
    ctx.strokeRect(x + 1.5, y + 1.5, CELL - 3, CELL - 3);
  }
  for (const cell of game.fruits) drawFruit(cell);
  drawSnake(game);
  if (overlay) drawOverlay(overlay);
}

function drawFruit(cell) {
  const [x, y] = cellXY(cell);
  const radius = CELL * 0.34, cx = x + CELL / 2, cy = y + CELL / 2 + 2;
  ctx.fillStyle = COLORS.fruit;
  ctx.beginPath(); ctx.arc(cx, cy, radius, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = COLORS.leaf;
  ctx.beginPath();
  ctx.ellipse(cx + radius / 3, cy - radius + 1, radius / 2, radius / 4, -0.4, 0, 2 * Math.PI);
  ctx.fill();
}

function drawSnake(game) {
  const body = game.snake, n = body.length, inset = CELL * 0.12;
  for (let i = n - 1; i >= 0; i--) {             // da cauda para a cabeça (cabeça por cima)
    const t = i / Math.max(n - 1, 1);              // degradê: cabeça clara, cauda escura
    const rgb = COLORS.head.map((h, k) => Math.round(h + (COLORS.tail[k] - h) * t));
    ctx.fillStyle = `rgb(${rgb})`;
    const [x, y] = cellXY(body[i]);
    ctx.beginPath();
    ctx.roundRect(x + inset, y + inset, CELL - 2 * inset, CELL - 2 * inset, inset * 2);
    ctx.fill();
    if (i + 1 < n) {                               // "ponte" até o próximo segmento
      const [x2, y2] = cellXY(body[i + 1]);
      const left = Math.min(x, x2), top = Math.min(y, y2);
      ctx.fillRect(left + inset, top + inset, Math.abs(x - x2) + CELL - 2 * inset, Math.abs(y - y2) + CELL - 2 * inset);
    }
  }
  // Olhos, perpendiculares à direção.
  const [dr, dc] = DELTAS[game.direction];
  const [hx, hy] = cellXY(game.head);
  const cx = hx + CELL / 2 + dc * CELL * 0.15, cy = hy + CELL / 2 + dr * CELL * 0.15;
  for (const side of [-1, 1]) {
    const ex = cx + side * dr * CELL * 0.18, ey = cy + side * dc * CELL * 0.18;
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(ex, ey, CELL * 0.11, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#141428";
    ctx.beginPath(); ctx.arc(ex + dc * 3, ey + dr * 3, CELL * 0.05, 0, 2 * Math.PI); ctx.fill();
  }
}

function drawOverlay({ title, subtitle }) {
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.font = "700 34px system-ui, sans-serif";
  ctx.fillText(title, canvas.width / 2, canvas.height / 2 - 6);
  ctx.font = "18px system-ui, sans-serif";
  ctx.fillText(subtitle, canvas.width / 2, canvas.height / 2 + 26);
}

// ---------------------------------------------------------------- a rede
const models = {};   // cache: cada .onnx é baixado uma vez só

async function loadModel(name) {
  if (!models[name]) models[name] = await ort.InferenceSession.create(`models/ppo_${name}.onnx`);
  return models[name];
}

/** Probabilidades das 4 ações (softmax das notas). Igual ao PPO: joga a maior. */
async function think(session, game) {
  const obs = observe(game);
  const out = await session.run({ obs: new ort.Tensor("float32", obs, [1, 18]) });
  const logits = Array.from(out.logits.data);
  const max = Math.max(...logits);
  const exp = logits.map((v) => Math.exp(v - max));   // "- max" evita estourar o exp
  const sum = exp.reduce((a, b) => a + b, 0);
  return { obs, probs: exp.map((v) => v / sum), action: logits.indexOf(max) };
}

// ---------------------------------------------------------------- estado da página
const $ = (id) => document.getElementById(id);
const END_TEXT = {
  wall: "Bateu na parede", body: "Bateu no próprio corpo", obstacle: "Bateu num obstáculo",
  timeout: "200 passos sem comer", win: "ZEROU O JOGO!",
};
const ARROWS = ["↑ cima", "→ direita", "↓ baixo", "← esquerda"];

let mode = "ai";
let game = newGame();
let paused = false;
let best = 0;
let timer = null;
let humanQueue = [];        // teclas apertadas entre um passo e outro (como no play.py)
let humanStarted = false;

function newGame() {
  // Seed sorteada: cada partida é diferente (veja a nota sobre o sorteio em snake.js).
  return new SnakeGame(Math.floor(Math.random() * 2 ** 31));
}

function updateHud() {
  $("fruits").textContent = game.fruitsCollected;
  $("obstacles").textContent = game.obstaclesGenerated;
  best = Math.max(best, game.fruitsCollected);
  $("best").textContent = best;
}

function showThoughts({ obs, probs, action }) {
  $("probs").innerHTML = probs.map((p, d) => `
    <div class="prob${d === action ? " chosen" : ""}">
      <span>${ARROWS[d]}</span>
      <span class="track"><span class="fill" style="width:${(p * 100).toFixed(1)}%"></span></span>
      <span class="pct">${(p * 100).toFixed(0)}%</span>
    </div>`).join("");
  const fmt = (v) => `${Math.round(v * 100)}%`;
  [0, 1, 2].forEach((k) => {
    $(`f${k}`).textContent = obs[k] ? "sim" : "não";
    $(`f${k}`).className = obs[k] ? "bad" : "";
    $(`f${11 + k}`).textContent = fmt(obs[11 + k]);
    $(`f${14 + k}`).textContent = obs[14 + k] ? "sim" : "não";
    $(`f${14 + k}`).className = obs[14 + k] ? "" : "bad";
  });
}

function finish() {
  updateHud();
  const subtitle = `${game.fruitsCollected} frutas em ${game.steps} passos`;
  draw(game, { title: END_TEXT[game.endReason], subtitle });
  $("status").textContent = `${END_TEXT[game.endReason]} — ${subtitle}.`;
}

// ---------------------------------------------------------------- laço da IA
let busy = false;
async function aiTick() {
  if (busy || paused || mode !== "ai") return;
  busy = true;
  try {
    if (game.gameOver) return;
    const session = await loadModel($("model").value);
    const thought = await think(session, game);
    showThoughts(thought);
    game.step(thought.action);
    draw(game);
    updateHud();
    if (game.gameOver) {
      finish();
      if ($("auto").checked) setTimeout(() => { if (mode === "ai") { game = newGame(); draw(game); } }, 1800);
    }
  } finally {
    busy = false;
  }
}

function startAi() {
  clearInterval(timer);
  timer = setInterval(aiTick, 1000 / Number($("speed").value));
}

// ---------------------------------------------------------------- modo humano
const KEYS = {
  ArrowUp: UP, KeyW: UP, ArrowRight: RIGHT, KeyD: RIGHT,
  ArrowDown: DOWN, KeyS: DOWN, ArrowLeft: LEFT, KeyA: LEFT,
};

function humanInput(direction) {
  if (mode !== "human" || game.gameOver) return;
  humanStarted = true;
  // Ignora repetição e meia-volta já na fila, para a cobra não "engolir" um comando.
  const last = humanQueue.length ? humanQueue[humanQueue.length - 1] : game.direction;
  if (direction !== last && direction !== OPPOSITE[last] && humanQueue.length < 3) humanQueue.push(direction);
}

function humanTick() {
  if (mode !== "human" || paused || !humanStarted || game.gameOver) return;
  game.step(humanQueue.length ? humanQueue.shift() : null);
  draw(game);
  updateHud();
  if (game.gameOver) finish();
}

function startHuman() {
  clearInterval(timer);
  game = newGame();
  humanQueue = [];
  humanStarted = false;
  draw(game, { title: "Sua vez", subtitle: "aperte uma seta para começar" });
  updateHud();
  $("status").textContent = "Setas / WASD para jogar.";
  timer = setInterval(humanTick, 1000 / 7);     // 7 passos/s, como no play.py
}

document.addEventListener("keydown", (e) => {
  if (mode !== "human") return;
  if (e.code === "Space") { paused = !paused; e.preventDefault(); return; }
  if (e.code in KEYS) { humanInput(KEYS[e.code]); e.preventDefault(); }
});

// Deslizar o dedo no celular: a direção é o eixo em que o dedo andou mais.
let touchStart = null;
canvas.addEventListener("touchstart", (e) => { touchStart = [e.touches[0].clientX, e.touches[0].clientY]; }, { passive: true });
canvas.addEventListener("touchend", (e) => {
  if (!touchStart) return;
  const dx = e.changedTouches[0].clientX - touchStart[0], dy = e.changedTouches[0].clientY - touchStart[1];
  touchStart = null;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 20) return;
  humanInput(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? RIGHT : LEFT) : (dy > 0 ? DOWN : UP));
});

// ---------------------------------------------------------------- botões
function setMode(next) {
  mode = next;
  paused = false;
  $("pause").textContent = "Pausar";
  $("mode-ai").setAttribute("aria-selected", next === "ai");
  $("mode-human").setAttribute("aria-selected", next === "human");
  $("mode-ai").classList.toggle("active", next === "ai");
  $("mode-human").classList.toggle("active", next === "human");
  $("ai-controls").hidden = next !== "ai";
  $("human-controls").hidden = next !== "human";
  if (next === "ai") {
    game = newGame();
    draw(game);
    $("status").textContent = "A rede está jogando.";
    startAi();
  } else {
    startHuman();
  }
}

$("mode-ai").onclick = () => setMode("ai");
$("mode-human").onclick = () => setMode("human");
$("restart").onclick = () => { game = newGame(); draw(game); updateHud(); $("status").textContent = "A rede está jogando."; };
$("restart-human").onclick = startHuman;
$("pause").onclick = () => { paused = !paused; $("pause").textContent = paused ? "Continuar" : "Pausar"; };
$("speed").oninput = () => { $("speed-label").textContent = $("speed").value; if (mode === "ai") startAi(); };
$("model").onchange = () => { game = newGame(); draw(game); updateHud(); };

// ---------------------------------------------------------------- início
draw(game);
// Os arquivos .wasm do onnxruntime vêm do mesmo CDN do script.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
loadModel("final")
  .then(() => { $("status").textContent = "A rede está jogando."; startAi(); })
  .catch((err) => { $("status").textContent = `Não consegui carregar a rede: ${err.message}`; });
