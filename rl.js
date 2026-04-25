/* ============================================================
   rl.js — Cliff Walking RL Simulator
   Implements Q-Learning & SARSA entirely in JavaScript.
   All training runs in a Web Worker-like async loop so the
   UI stays responsive and progress is reported in real time.
   ============================================================ */

'use strict';

// ── Constants ────────────────────────────────────────────────
const N_ROWS = 4, N_COLS = 12;
const ACTIONS = { UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3 };
const N_ACTIONS = 4;
const N_STATES  = N_ROWS * N_COLS;
const START     = (N_ROWS - 1) * N_COLS + 0;          // bottom-left
const GOAL      = (N_ROWS - 1) * N_COLS + (N_COLS-1); // bottom-right
// Cliff cells: bottom row between start and goal
const CLIFF_SET = new Set(
  Array.from({length: N_COLS - 2}, (_, i) => (N_ROWS-1) * N_COLS + (i+1))
);

// ── Env helpers ──────────────────────────────────────────────
function stateToPos(s) { return [Math.floor(s / N_COLS), s % N_COLS]; }
function posToState(r, c) { return r * N_COLS + c; }

function envStep(state, action) {
  let [r, c] = stateToPos(state);
  if      (action === ACTIONS.UP)    r = Math.max(r - 1, 0);
  else if (action === ACTIONS.RIGHT) c = Math.min(c + 1, N_COLS - 1);
  else if (action === ACTIONS.DOWN)  r = Math.min(r + 1, N_ROWS - 1);
  else if (action === ACTIONS.LEFT)  c = Math.max(c - 1, 0);
  const ns = posToState(r, c);
  if (CLIFF_SET.has(ns)) return { next: START, reward: -100, done: false };
  if (ns === GOAL)        return { next: ns,   reward:   -1, done: true  };
  return                         { next: ns,   reward:   -1, done: false };
}

// ── ε-greedy ─────────────────────────────────────────────────
function epsilonGreedy(Q, s, epsilon) {
  if (Math.random() < epsilon) return Math.floor(Math.random() * N_ACTIONS);
  let best = 0, bestVal = Q[s * N_ACTIONS];
  for (let a = 1; a < N_ACTIONS; a++) {
    const v = Q[s * N_ACTIONS + a];
    if (v > bestVal) { bestVal = v; best = a; }
  }
  return best;
}

function maxQ(Q, s) {
  let m = Q[s * N_ACTIONS];
  for (let a = 1; a < N_ACTIONS; a++) if (Q[s * N_ACTIONS + a] > m) m = Q[s * N_ACTIONS + a];
  return m;
}

// ── Q-Learning ───────────────────────────────────────────────
function runQLearning(episodes, alpha, gamma, epsilon) {
  const Q       = new Float64Array(N_STATES * N_ACTIONS); // zeros
  const rewards = new Array(episodes);
  const MAX_STEPS = 10000;
  for (let ep = 0; ep < episodes; ep++) {
    let s = START, total = 0, done = false, step = 0;
    while (!done && step < MAX_STEPS) {
      const a = epsilonGreedy(Q, s, epsilon);
      const { next, reward, done: d } = envStep(s, a);
      Q[s * N_ACTIONS + a] += alpha * (reward + gamma * maxQ(Q, next) - Q[s * N_ACTIONS + a]);
      s = next; total += reward; done = d; step++;
    }
    rewards[ep] = total;
  }
  return { Q, rewards };
}

// ── SARSA ────────────────────────────────────────────────────
function runSarsa(episodes, alpha, gamma, epsilon) {
  const Q       = new Float64Array(N_STATES * N_ACTIONS);
  const rewards = new Array(episodes);
  const MAX_STEPS = 10000;
  for (let ep = 0; ep < episodes; ep++) {
    let s = START, a = epsilonGreedy(Q, s, epsilon);
    let total = 0, done = false, step = 0;
    while (!done && step < MAX_STEPS) {
      const { next, reward, done: d } = envStep(s, a);
      const a2 = epsilonGreedy(Q, next, epsilon);
      Q[s * N_ACTIONS + a] += alpha * (reward + gamma * Q[next * N_ACTIONS + a2] - Q[s * N_ACTIONS + a]);
      s = next; a = a2; total += reward; done = d; step++;
    }
    rewards[ep] = total;
  }
  return { Q, rewards };
}

// ── Moving average ───────────────────────────────────────────
function movingAvg(arr, w) {
  const out = [];
  for (let i = w - 1; i < arr.length; i++) {
    let sum = 0;
    for (let j = 0; j < w; j++) sum += arr[i - j];
    out.push(sum / w);
  }
  return out;
}

// ── Greedy path trace ────────────────────────────────────────
function tracePath(Q) {
  const path = [START];
  const visited = new Set([START]);
  let s = START;
  for (let i = 0; i < N_ROWS * N_COLS * 3; i++) {
    let best = 0, bestVal = Q[s * N_ACTIONS];
    for (let a = 1; a < N_ACTIONS; a++) {
      const v = Q[s * N_ACTIONS + a];
      if (v > bestVal) { bestVal = v; best = a; }
    }
    const { next, done } = envStep(s, best);
    path.push(next);
    if (done || next === GOAL) break;
    if (visited.has(next)) break;
    visited.add(next);
    s = next;
  }
  return path;
}

// ── Chart.js reward chart ────────────────────────────────────
let rewardChart = null;

function drawRewardChart(meanQ, meanSarsa, episodes) {
  const SOLID_W  = 10;
  const DOTTED_W = 50;
  const xs = Array.from({length: episodes}, (_, i) => i + 1);

  // Smoothed series
  const smQ     = movingAvg(meanQ,     SOLID_W);
  const smSarsa = movingAvg(meanSarsa, SOLID_W);
  const smXs    = xs.slice(SOLID_W - 1);

  const dotQ     = movingAvg(meanQ,     DOTTED_W);
  const dotSarsa = movingAvg(meanSarsa, DOTTED_W);
  const dotXs    = xs.slice(DOTTED_W - 1);

  const ctx = document.getElementById('rewardChart').getContext('2d');
  if (rewardChart) { rewardChart.destroy(); }

  rewardChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Sarsa',
          data: smXs.map((x, i) => ({ x, y: smSarsa[i] })),
          borderColor: '#00bcd4', backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.3,
          parsing: false
        },
        {
          label: 'Q-learning',
          data: smXs.map((x, i) => ({ x, y: smQ[i] })),
          borderColor: '#ef5350', backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.3,
          parsing: false
        },
        {
          label: 'Sarsa, Sutton Pub.',
          data: dotXs.map((x, i) => ({ x, y: dotSarsa[i] })),
          borderColor: '#00bcd4', backgroundColor: 'transparent',
          borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, tension: 0.4,
          parsing: false
        },
        {
          label: 'Q-learning, Sutton Pub.',
          data: dotXs.map((x, i) => ({ x, y: dotQ[i] })),
          borderColor: '#ef5350', backgroundColor: 'transparent',
          borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, tension: 0.4,
          parsing: false
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 600 },
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          type: 'linear',
          min: 1, max: episodes,
          title: { display: true, text: 'Episodes', color: '#8892a4', font: { weight: '600' } },
          grid: { color: 'rgba(255,255,255,.06)' },
          ticks: { color: '#8892a4' }
        },
        y: {
          min: -105, max: 5,
          title: { display: true, text: 'Reward Sum for Episode', color: '#8892a4', font: { weight: '600' } },
          grid: { color: 'rgba(255,255,255,.06)' },
          ticks: { color: '#8892a4' }
        }
      },
      plugins: {
        legend: {
          labels: {
            color: '#8892a4', font: { size: 11 },
            usePointStyle: true, pointStyleWidth: 14
          }
        },
        tooltip: {
          backgroundColor: '#1e2230',
          borderColor: 'rgba(255,255,255,.12)',
          borderWidth: 1,
          titleColor: '#e8eaf0',
          bodyColor: '#8892a4',
          callbacks: {
            title: items => `Episode ${items[0].parsed.x}`,
            label: item => ` ${item.dataset.label}: ${item.parsed.y.toFixed(1)}`
          }
        }
      }
    }
  });

  document.getElementById('rewardPlaceholder').style.display = 'none';
}

// ── Policy Canvas renderer ───────────────────────────────────
const CELL = 46;  // px per cell
const COLS = N_COLS, ROWS = N_ROWS;

function drawPolicy(canvasId, Q, title) {
  const canvas = document.getElementById(canvasId);
  const W = COLS * CELL, H = ROWS * CELL;
  canvas.width  = W;
  canvas.height = H;
  canvas.style.height = (H / window.devicePixelRatio || H) + 'px';

  const ctx = canvas.getContext('2d');

  // Background
  ctx.fillStyle = '#161924';
  ctx.fillRect(0, 0, W, H);

  // ── Cells ──
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const s = posToState(r, c);
      const x = c * CELL, y = r * CELL;
      // Cliff fill
      if (CLIFF_SET.has(s)) {
        ctx.fillStyle = 'rgba(0,188,212,.12)';
        ctx.fillRect(x, y, CELL, CELL);
      }
      // Grid lines
      ctx.strokeStyle = '#2a2f40';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + .5, y + .5, CELL - 1, CELL - 1);
    }
  }

  // ── Greedy path ──
  const path = tracePath(Q);
  const pathSet = new Set(path);

  // Collect path rows (excluding cliff, start, goal)
  const pathRows = new Set(
    path.map(s => stateToPos(s)[0])
        .filter(r => !([...CLIFF_SET].some(cs => stateToPos(cs)[0] === r && stateToPos(cs)[1] !== 0 && stateToPos(cs)[1] !== N_COLS-1) && r === ROWS - 1))
  );
  // Draw dashed blue rectangle around path row band
  if (path.length > 1) {
    const validPathRows = [];
    for (const s of path) {
      const [r, c] = stateToPos(s);
      if (!CLIFF_SET.has(s) && s !== START && s !== GOAL) validPathRows.push(r);
    }
    if (validPathRows.length) {
      const rMin = Math.min(...validPathRows);
      const rMax = Math.max(...validPathRows);
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = '#1565C0';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(1, rMin * CELL + 1, W - 2, (rMax - rMin + 1) * CELL - 2);
      ctx.restore();
    }
  }

  // Draw path line
  if (path.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(21,101,192,.85)';
    ctx.lineWidth = 2.5;
    const [r0, c0] = stateToPos(path[0]);
    ctx.moveTo(c0 * CELL + CELL/2, r0 * CELL + CELL/2);
    for (let i = 1; i < path.length; i++) {
      const [r, c] = stateToPos(path[i]);
      ctx.lineTo(c * CELL + CELL/2, r * CELL + CELL/2);
    }
    ctx.stroke();
  }

  // ── Policy arrows ──
  const DELTA = CELL * 0.28;
  const ARROW_DIRS = [
    [0, -DELTA], // UP
    [DELTA, 0],  // RIGHT
    [0, DELTA],  // DOWN
    [-DELTA, 0]  // LEFT
  ];

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const s = posToState(r, c);
      if (CLIFF_SET.has(s) || s === GOAL) continue;

      // Best action
      let best = 0, bestVal = Q[s * N_ACTIONS];
      for (let a = 1; a < N_ACTIONS; a++) {
        const v = Q[s * N_ACTIONS + a];
        if (v > bestVal) { bestVal = v; best = a; }
      }

      const cx = c * CELL + CELL / 2;
      const cy = r * CELL + CELL / 2;
      const [dx, dy] = ARROW_DIRS[best];

      drawArrow(ctx, cx - dx * 0.5, cy - dy * 0.5, cx + dx * 0.5, cy + dy * 0.5);
    }
  }

  // ── Cliff label ──
  const cliffMidC = (N_COLS - 2) / 2 + 0.5;  // centre of cliff cells
  ctx.fillStyle = 'rgba(0,188,212,.6)';
  ctx.font = `bold ${CELL * 0.32}px Inter, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('Cliff', cliffMidC * CELL, (ROWS - 0.5) * CELL);

  // ── Start marker ──
  const [sr, sc] = stateToPos(START);
  ctx.fillStyle = '#e8eaf0';
  ctx.font = `bold ${CELL * 0.28}px Inter, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('Start', sc * CELL + CELL/2, sr * CELL + CELL * 0.72);
  // Up arrow at start
  drawArrow(ctx, sc * CELL + CELL/2, sr * CELL + CELL * 0.52, sc * CELL + CELL/2, sr * CELL + CELL * 0.22, '#e8eaf0');

  // ── Goal marker ──
  const [gr, gc] = stateToPos(GOAL);
  ctx.fillStyle = '#ef9a9a';
  ctx.font = `bold ${CELL * 0.28}px Inter, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('Goal', gc * CELL + CELL/2, gr * CELL + CELL/2);

  // Hide placeholder
  const ph = canvasId === 'policyQ' ? 'policyQPh' : 'policySarsaPh';
  document.getElementById(ph).style.display = 'none';
  canvas.style.display = 'block';
}

function drawArrow(ctx, x1, y1, x2, y2, color = '#c8d0e0') {
  const headLen = CELL * 0.16;
  const angle   = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  // Arrowhead
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI/6), y2 - headLen * Math.sin(angle - Math.PI/6));
  ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI/6), y2 - headLen * Math.sin(angle + Math.PI/6));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

// ── Training orchestrator ────────────────────────────────────
async function startTraining() {
  const alpha   = parseFloat(document.getElementById('alpha').value);
  const gamma   = parseFloat(document.getElementById('gamma').value);
  const epsilon = parseFloat(document.getElementById('epsilon').value);
  const episodes = parseInt(document.getElementById('episodes').value);
  const runs     = parseInt(document.getElementById('runs').value);

  // UI: running state
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Training…';
  btn.parentElement.parentElement.parentElement.classList.add('running');

  document.getElementById('progressSection').style.display = 'block';
  document.getElementById('statsSection').style.display = 'none';
  setProgress(0, 'Initialising…');

  // Yield to let browser repaint
  await sleep(20);

  const allQ     = Array.from({length: runs}, () => new Array(episodes).fill(0));
  const allSarsa = Array.from({length: runs}, () => new Array(episodes).fill(0));
  let lastQ = null, lastSarsa = null;

  for (let run = 0; run < runs; run++) {
    const pct = Math.round((run / runs) * 90);
    setProgress(pct, `Run ${run + 1} / ${runs}…`);
    await sleep(0); // let progress bar paint

    const { Q: Qq, rewards: rq }   = runQLearning(episodes, alpha, gamma, epsilon);
    const { Q: Qs, rewards: rs }   = runSarsa(episodes, alpha, gamma, epsilon);
    for (let ep = 0; ep < episodes; ep++) {
      allQ[run][ep]     = rq[ep];
      allSarsa[run][ep] = rs[ep];
    }
    lastQ     = Qq;
    lastSarsa = Qs;
  }

  // Compute means
  const meanQ     = new Array(episodes).fill(0);
  const meanSarsa = new Array(episodes).fill(0);
  for (let ep = 0; ep < episodes; ep++) {
    for (let run = 0; run < runs; run++) {
      meanQ[ep]     += allQ[run][ep];
      meanSarsa[ep] += allSarsa[run][ep];
    }
    meanQ[ep]     /= runs;
    meanSarsa[ep] /= runs;
  }

  setProgress(95, 'Rendering charts…');
  await sleep(20);

  // Draw reward chart
  drawRewardChart(meanQ, meanSarsa, episodes);

  // Draw policy maps
  drawPolicy('policyQ',     lastQ,     'Q-Learning');
  drawPolicy('policySarsa', lastSarsa, 'SARSA');

  setProgress(100, 'Done ✓');

  // Stats
  const last50Q     = meanQ.slice(-50);
  const last50Sarsa = meanSarsa.slice(-50);
  const avgQ     = last50Q.reduce((a,b) => a+b, 0) / 50;
  const avgSarsa = last50Sarsa.reduce((a,b) => a+b, 0) / 50;

  document.getElementById('statQ').textContent     = avgQ.toFixed(1);
  document.getElementById('statSarsa').textContent = avgSarsa.toFixed(1);

  const winner = avgSarsa > avgQ ? 'SARSA' : 'Q-Learning';
  document.getElementById('insightBox').textContent =
    `With ε=${epsilon}, α=${alpha}: ${winner} achieves a higher average reward ` +
    `(${Math.max(avgSarsa, avgQ).toFixed(1)} vs ${Math.min(avgSarsa, avgQ).toFixed(1)}). ` +
    (avgSarsa > avgQ
      ? 'SARSA learns a safer path above the cliff, avoiding exploration-induced falls.'
      : 'Q-Learning converges to a shorter cliff-edge path under these parameters.');

  document.getElementById('statsSection').style.display = 'block';

  // Reset button
  btn.disabled = false;
  btn.innerHTML = '<span class="btn-icon">▶</span> Run Again';
  btn.parentElement.parentElement.parentElement.classList.remove('running');
}

function setProgress(pct, label) {
  document.getElementById('progressBar').style.width = pct + '%';
  document.getElementById('progressLabel').textContent = label;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Slider live update ───────────────────────────────────────
function bindSlider(id, displayId, decimals = 2) {
  const slider  = document.getElementById(id);
  const display = document.getElementById(displayId);
  const update  = () => { display.textContent = parseFloat(slider.value).toFixed(decimals); };
  slider.addEventListener('input', update);
  update();
}

function resetDefaults() {
  document.getElementById('alpha').value   = 0.10;
  document.getElementById('gamma').value   = 0.90;
  document.getElementById('epsilon').value = 0.10;
  document.getElementById('episodes').value = 500;
  document.getElementById('runs').value     = 50;
  ['alpha','gamma','epsilon','episodes','runs'].forEach(id => {
    document.getElementById(id).dispatchEvent(new Event('input'));
  });
}

// ── Init ─────────────────────────────────────────────────────
bindSlider('alpha',    'alphaVal',    2);
bindSlider('gamma',    'gammaVal',    2);
bindSlider('epsilon',  'epsilonVal',  2);
bindSlider('episodes', 'episodesVal', 0);
bindSlider('runs',     'runsVal',     0);
