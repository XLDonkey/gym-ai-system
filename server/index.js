/**
 * XL Fitness AI Overseer — Mac Mini Proxy Server
 * Bridges the public dashboard (manus.space) to local Pi nodes
 * Runs on mac-mini.local:3001
 */

const express = require('express');
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

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

// Root
app.get('/', (req, res) => {
  res.json({ 
    service: 'XL Fitness Overseer Mac Mini Server',
    version: '1.0.0',
    endpoints: ['/api/health', '/api/nodes', '/api/refresh', '/api/snapshot/:id']
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`XL Fitness Overseer Server running on port ${PORT}`);
  console.log(`Local: http://localhost:${PORT}`);
  console.log(`Network: http://mac-mini.local:${PORT}`);
});
