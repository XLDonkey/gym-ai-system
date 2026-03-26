/**
 * XL Fitness AI Overseer — Mac Mini Proxy Server
 * Bridges the public dashboard (manus.space) to local Pi nodes
 * Runs on mac-mini.local:3001
 */

const express = require('express');
const http = require('http');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { randomUUID } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3001;
const NODES_FILE = path.join(__dirname, 'nodes.json');
const CACHE_TTL_MS = 30000; // 30 second cache

// CORS — allow dashboard from any origin on the local network
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

app.use(express.json());

// Load nodes config
function loadNodes() {
  return JSON.parse(fs.readFileSync(NODES_FILE, 'utf8'));
}

// Cache for node status
const statusCache = new Map();

// Fetch status from a single Pi
function fetchPiStatus(node) {
  return new Promise((resolve) => {
    const options = {
      hostname: node.ip,
      port: node.port,
      path: '/status',
      method: 'GET',
      timeout: 5000,
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        try {
          const status = JSON.parse(data);
          resolve({
            ...node,
            online: true,
            ...status,
            last_fetched: Date.now(),
          });
        } catch {
          resolve({ ...node, online: false, last_fetched: Date.now() });
        }
      });
    });

    req.on('error', () => resolve({ ...node, online: false, last_fetched: Date.now() }));
    req.on('timeout', () => { req.destroy(); resolve({ ...node, online: false, last_fetched: Date.now() }); });
    req.end();
  });
}

// Refresh all nodes
async function refreshAllNodes() {
  const nodes = loadNodes();
  const results = await Promise.all(nodes.map(fetchPiStatus));
  results.forEach((r) => statusCache.set(r.id, r));
  return results;
}

// Auto-refresh every 30 seconds
setInterval(refreshAllNodes, 30000);
refreshAllNodes(); // Initial fetch on startup

// ── Routes ────────────────────────────────────────────────────────────────────

// GET /api/nodes — return cached status for all Pi nodes
app.get('/api/nodes', (req, res) => {
  const nodes = loadNodes();
  const result = nodes.map((n) => statusCache.get(n.id) || { ...n, online: false, last_fetched: null });
  res.json(result);
});

// POST /api/refresh — force refresh all nodes and return fresh data
app.post('/api/refresh', async (req, res) => {
  const results = await refreshAllNodes();
  res.json(results);
});

// GET /api/snapshot/:id — proxy snapshot JPEG from Pi
app.get('/api/snapshot/:id', (req, res) => {
  const nodes = loadNodes();
  const node = nodes.find((n) => n.id === req.params.id);
  if (!node) return res.status(404).json({ error: 'Node not found' });

  const options = {
    hostname: node.ip,
    port: node.port,
    path: '/snapshot',
    method: 'GET',
    timeout: 8000,
  };

  const piReq = http.request(options, (piRes) => {
    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Cache-Control', 'no-cache');
    piRes.pipe(res);
  });

  piReq.on('error', () => res.status(502).json({ error: 'Pi unreachable' }));
  piReq.on('timeout', () => { piReq.destroy(); res.status(504).json({ error: 'Pi timeout' }); });
  piReq.end();
});

// GET /api/health — server health check
app.get('/api/health', (req, res) => {
  const nodes = loadNodes();
  const uptime = process.uptime();
  res.json({
    status: 'ok',
    nodes: nodes.length,
    uptime_s: Math.floor(uptime),
    hostname: os.hostname(),
    port: PORT,
    timestamp: Date.now(),
  });
});

// ── Training API ──────────────────────────────────────────────────────────────

// In-memory job store (survives until server restart)
const trainJobs = new Map();

// Resolve paths relative to this repo
const REPO_ROOT = path.resolve(__dirname, '..');
const TRAIN_SCRIPT = path.join(REPO_ROOT, 'train', 'overseer_train.py');
const MODELS_DIR = path.join(REPO_ROOT, 'models');

/**
 * POST /api/train
 * Body: { "drive_folder": "1KNDC4w...", "output_name": "overseer_v1" }
 * Returns: { "jobId": "uuid", "status": "running" }
 */
app.post('/api/train', (req, res) => {
  const { drive_folder, output_name = 'overseer_v1' } = req.body || {};

  if (!drive_folder) {
    return res.status(400).json({ error: 'drive_folder is required' });
  }

  const jobId = randomUUID();
  const outputPath = path.join(MODELS_DIR, `${output_name}.onnx`);
  const logPath = path.join(MODELS_DIR, `${output_name}_${jobId.slice(0, 8)}.log`);

  fs.mkdirSync(MODELS_DIR, { recursive: true });

  const job = {
    jobId,
    status: 'running',
    drive_folder,
    output_name,
    output_path: outputPath,
    log_path: logPath,
    started_at: new Date().toISOString(),
    finished_at: null,
    accuracy: null,
    error: null,
  };
  trainJobs.set(jobId, job);

  console.log(`[train] Starting job ${jobId} — drive_folder=${drive_folder} output=${outputPath}`);

  // Launch training subprocess
  const logStream = fs.createWriteStream(logPath, { flags: 'a' });
  const proc = spawn('python3', [
    TRAIN_SCRIPT,
    '--drive-folder', drive_folder,
    '--output', outputPath,
  ], {
    cwd: REPO_ROOT,
    env: { ...process.env },
  });

  proc.stdout.pipe(logStream);
  proc.stderr.pipe(logStream);

  // Try to parse accuracy from stdout
  proc.stdout.on('data', (chunk) => {
    const text = chunk.toString();
    // Look for "Best validation accuracy: 0.923"
    const match = text.match(/Best validation accuracy:\s*([\d.]+)/);
    if (match) {
      job.accuracy = parseFloat(match[1]);
    }
  });

  proc.on('close', (code) => {
    job.finished_at = new Date().toISOString();
    if (code === 0) {
      job.status = 'complete';
      console.log(`[train] Job ${jobId} complete — accuracy=${job.accuracy}`);
    } else {
      job.status = 'failed';
      job.error = `Process exited with code ${code}`;
      console.log(`[train] Job ${jobId} failed (exit ${code})`);
    }
    logStream.end();
  });

  proc.on('error', (err) => {
    job.status = 'failed';
    job.error = err.message;
    job.finished_at = new Date().toISOString();
    console.log(`[train] Job ${jobId} error: ${err.message}`);
    logStream.end();
  });

  res.json({ jobId, status: 'running', message: `Training started. Poll GET /api/train/${jobId} for status.` });
});

/**
 * GET /api/train/:jobId
 * Returns job status, accuracy when complete, and log tail.
 */
app.get('/api/train/:jobId', (req, res) => {
  const job = trainJobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }

  // Include last 20 lines of log if available
  let log_tail = null;
  try {
    const log = fs.readFileSync(job.log_path, 'utf8');
    log_tail = log.split('\n').slice(-20).join('\n');
  } catch (_) { /* log not yet written */ }

  res.json({
    jobId: job.jobId,
    status: job.status,
    drive_folder: job.drive_folder,
    output_name: job.output_name,
    output_path: job.output_path,
    started_at: job.started_at,
    finished_at: job.finished_at,
    accuracy: job.accuracy,
    error: job.error,
    log_tail,
  });
});

/**
 * GET /api/train
 * List all training jobs.
 */
app.get('/api/train', (req, res) => {
  const jobs = Array.from(trainJobs.values()).map(({ jobId, status, output_name, started_at, finished_at, accuracy }) => ({
    jobId, status, output_name, started_at, finished_at, accuracy,
  }));
  res.json(jobs);
});

// Root
app.get('/', (req, res) => {
  res.json({ 
    service: 'XL Fitness Overseer Mac Mini Server',
    version: '1.1.0',
    endpoints: ['/api/health', '/api/nodes', '/api/refresh', '/api/snapshot/:id', '/api/train']
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`XL Fitness Overseer Server running on port ${PORT}`);
  console.log(`Local: http://localhost:${PORT}`);
  console.log(`Network: http://mac-mini.local:${PORT}`);
});
