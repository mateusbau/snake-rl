// Motor do jogo em JavaScript: tradução fiel de snake/env.py (Fase 8).
//
// As regras são as MESMAS do Python (ordem do passo, obstáculos, becos, vitória,
// limite de 200 passos sem comer). O teste web/tests/parity.test.js confere isso
// contra estados gerados pelo Python.
//
// A única diferença é o sorteio: o Python usa o gerador do numpy (PCG64), que não
// existe no navegador. Aqui usamos o mulberry32, um gerador pequeno e conhecido.
// Por isso a seed 42 no navegador NÃO dá a mesma partida que a seed 42 no Python;
// as regras são as mesmas, só os sorteios diferem.
//
// Células são arrays [linha, coluna]. Para guardar em Set, viram texto "r,c"
// (um Set de JS compara arrays por referência, não por valor).

export const UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3;
export const DELTAS = [[-1, 0], [0, 1], [1, 0], [0, -1]];
export const OPPOSITE = [DOWN, LEFT, UP, RIGHT];
export const WALL = "wall", BODY = "body", OBSTACLE = "obstacle", TIMEOUT = "timeout", WIN = "win";

export const CONFIG = {
  height: 9,
  width: 10,
  initialSnake: [[4, 3], [4, 2], [4, 1]],
  initialDirection: RIGHT,
  initialFruits: [[2, 4], [2, 8], [4, 6], [6, 4], [6, 8]],
  safetyZoneLength: 3,
  maxStepsWithoutFruit: 200,
};

export const key = ([r, c]) => `${r},${c}`;

/** Gerador de números aleatórios com seed (mulberry32). Devolve inteiros em [0, n). */
export function makeRng(seed) {
  let a = seed >>> 0;
  return (n) => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return Math.floor((((t ^ (t >>> 14)) >>> 0) / 4294967296) * n);
  };
}

export class SnakeGame {
  constructor(seed = 0, config = CONFIG) {
    this.config = config;
    this.seed = seed;
    this.rng = makeRng(seed);
    this.loadState({
      snake: config.initialSnake, direction: config.initialDirection,
      fruits: config.initialFruits, obstacles: [], fruitsCollected: 0,
    });
  }

  /** Monta um estado qualquer (usado pelo teste de paridade com o Python). */
  loadState({ snake, direction, fruits, obstacles = [], fruitsCollected = 0, stepsSinceFruit = 0 }) {
    this.snake = snake.map((c) => [...c]);       // cabeça no índice 0
    this.direction = direction;
    this.fruits = fruits.map((c) => [...c]);
    this.obstacles = obstacles.map((c) => [...c]);
    this.fruitsCollected = fruitsCollected;
    this.obstaclesGenerated = this.obstacles.length;
    this.steps = 0;
    this.stepsSinceFruit = stepsSinceFruit;
    this.gameOver = false;
    this.endReason = null;
  }

  get head() { return this.snake[0]; }

  inBounds([r, c]) { return r >= 0 && r < this.config.height && c >= 0 && c < this.config.width; }

  /** Avança um passo. Ordem: mover -> colisão -> comer -> repor fruta -> obstáculo (se ímpar) -> vitória. */
  step(action = null) {
    if (this.gameOver) throw new Error("A partida já acabou.");
    const result = { ateFruit: false, newFruit: null, newObstacle: null };
    if (action !== null && action !== OPPOSITE[this.direction]) this.direction = action;

    const [dr, dc] = DELTAS[this.direction];
    const newHead = [this.head[0] + dr, this.head[1] + dc];
    this.steps += 1;

    const eating = this.fruits.some((f) => f[0] === newHead[0] && f[1] === newHead[1]);
    // Se não come, a cauda sai junto: entrar na casa da cauda é permitido.
    const bodyToCheck = eating ? this.snake : this.snake.slice(0, -1);
    const has = (list, c) => list.some((x) => x[0] === c[0] && x[1] === c[1]);

    if (!this.inBounds(newHead)) return this.end(result, WALL);
    if (has(this.obstacles, newHead)) return this.end(result, OBSTACLE);
    if (has(bodyToCheck, newHead)) return this.end(result, BODY);

    this.snake.unshift(newHead);
    if (!eating) this.snake.pop();

    if (eating) {
      result.ateFruit = true;
      this.fruits = this.fruits.filter((f) => !(f[0] === newHead[0] && f[1] === newHead[1]));
      this.fruitsCollected += 1;
      this.stepsSinceFruit = 0;
      result.newFruit = this.spawnFruit();          // null se não há célula vazia
      if (this.fruitsCollected % 2 === 1) result.newObstacle = this.trySpawnObstacle();
      if (this.isWon()) return this.end(result, WIN);
    } else {
      this.stepsSinceFruit += 1;
      if (this.stepsSinceFruit >= this.config.maxStepsWithoutFruit) return this.end(result, TIMEOUT);
    }
    return result;
  }

  /** Zerou: cobra + obstáculos ocupam todas as células (nenhuma vazia, nenhuma fruta). */
  isWon() {
    return this.snake.length + this.obstacles.length === this.config.height * this.config.width;
  }

  end(result, reason) {
    this.gameOver = true;
    this.endReason = reason;
    return { ...result, gameOver: true, endReason: reason };
  }

  /** Células vazias em ordem fixa (linha a linha), como no Python. */
  emptyCells() {
    const occupied = new Set([...this.snake, ...this.fruits, ...this.obstacles].map(key));
    const cells = [];
    for (let r = 0; r < this.config.height; r++)
      for (let c = 0; c < this.config.width; c++)
        if (!occupied.has(`${r},${c}`)) cells.push([r, c]);
    return cells;
  }

  spawnFruit() {
    const cells = this.emptyCells();
    if (cells.length === 0) return null;
    const cell = cells[this.rng(cells.length)];
    this.fruits.push(cell);
    return cell;
  }

  /** As até 3 células à frente da cabeça (onde obstáculo não pode nascer). */
  safetyZone() {
    const [dr, dc] = DELTAS[this.direction];
    const zone = [];
    for (let k = 1; k <= this.config.safetyZoneLength; k++) {
      const cell = [this.head[0] + k * dr, this.head[1] + k * dc];
      if (this.inBounds(cell)) zone.push(cell);
    }
    return zone;
  }

  /** Onde um obstáculo pode nascer: vazia, longe de outro obstáculo (8 vizinhas),
   *  fora da zona de segurança e sem criar beco (toda célula livre com >= 2 saídas). */
  eligibleObstacleCells() {
    const { height, width } = this.config;
    const obstacles = new Set(this.obstacles.map(key));
    const protectedCells = new Set(this.safetyZone().map(key));
    const blocked = (r, c) => r < 0 || r >= height || c < 0 || c >= width || obstacles.has(`${r},${c}`);
    // Saídas de cada célula livre (só parede e obstáculo bloqueiam; cobra e fruta saem do lugar).
    const exits = new Map();
    for (let r = 0; r < height; r++)
      for (let c = 0; c < width; c++)
        if (!obstacles.has(`${r},${c}`))
          exits.set(`${r},${c}`, [[-1, 0], [1, 0], [0, -1], [0, 1]].filter(([dr, dc]) => !blocked(r + dr, c + dc)).length);

    return this.emptyCells().filter(([r, c]) => {
      if (protectedCells.has(`${r},${c}`)) return false;
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++)
          if ((dr || dc) && obstacles.has(`${r + dr},${c + dc}`)) return false;
      // Com obstáculo aqui, cada vizinha livre perde 1 saída: se tinha 2, vira beco.
      return ![[-1, 0], [1, 0], [0, -1], [0, 1]].some(([dr, dc]) => (exits.get(`${r + dr},${c + dc}`) ?? 99) <= 2);
    });
  }

  trySpawnObstacle() {
    const cells = this.eligibleObstacleCells();
    if (cells.length === 0) return null;
    const cell = cells[this.rng(cells.length)];
    this.obstacles.push(cell);
    this.obstaclesGenerated += 1;
    return cell;
  }

  /** Direções que não matam no próximo passo (sem a meia-volta). */
  safeDirections() {
    const safe = [];
    const obstacles = new Set(this.obstacles.map(key));
    const fruits = new Set(this.fruits.map(key));
    for (let d = 0; d < 4; d++) {
      if (d === OPPOSITE[this.direction]) continue;
      const cell = [this.head[0] + DELTAS[d][0], this.head[1] + DELTAS[d][1]];
      const body = fruits.has(key(cell)) ? this.snake : this.snake.slice(0, -1);
      if (this.inBounds(cell) && !obstacles.has(key(cell)) && !body.some((b) => key(b) === key(cell))) safe.push(d);
    }
    return safe;
  }
}
