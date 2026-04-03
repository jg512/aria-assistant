"""
dashboard.py — Lightweight web dashboard (Flask)
Accessible at http://raspberrypi.local:5000

Provides:
  - Live conversation history (Terminal style)
  - Text input (Execute commands)
  - System status (CPU, RAM, THERMAL, UPTIME)
  - Tool state (Music, Alarms, Todo, etc.)
  - Photo Frame mode (local photos from media/photos)
  - Visualizer (simulated spectrum based on state)
  - Interactive controls for music, Spotify, and home automation
"""

import logging
import os
import platform
import threading
import time
import queue
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Optional, List

logger = logging.getLogger(__name__)

# Dashboard state shared between Flask and the main loop
_history: Deque[dict] = deque(maxlen=50)
_status: dict = {"state": "idle", "last_heard": None, "last_said": None}
_tools_ref: dict = {}
_command_queue: queue.Queue = queue.Queue()
_history_lock = threading.Lock()
_photo_dir: Optional[Path] = None


def push_user(text: str):
    with _history_lock:
        _history.append({"role": "user", "text": text, "ts": datetime.now().strftime("%H:%M:%S")})
    _status["last_heard"] = text


def push_assistant(text: str):
    with _history_lock:
        _history.append({"role": "assistant", "text": text, "ts": datetime.now().strftime("%H:%M:%S")})
    _status["last_said"] = text


def set_state(state: str):
    _status["state"] = state


def get_next_command(timeout: float = 0.1) -> Optional[str]:
    """Poll for the next command sent from the web dashboard."""
    try:
        return _command_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def _build_app():
    try:
        from flask import Flask, jsonify, render_template_string, request, send_from_directory
    except ImportError:
        return None

    app = Flask(__name__)
    app.logger.setLevel(logging.ERROR)   # suppress Flask request logs

    HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aria | System Core</title>
<style>
  :root { 
    --bg:#05070a; 
    --surface:#0d1117; 
    --accent:#00f2ff; 
    --accent-dim:#008899;
    --text:#00f2ff; 
    --muted:#005566; 
    --user:#00f2ff; 
    --aria:#ffffff;
    --border: 1px solid #00f2ff33;
    --glow: 0 0 10px rgba(0, 242, 255, 0.3);
  }
  
  @font-face {
    font-family: 'Terminus';
    src: local('Courier New'), local('monospace');
  }

  * { box-sizing:border-box; margin:0; padding:0; }
  
  body { 
    background:var(--bg); 
    color:var(--text); 
    font-family:'Terminus', monospace; 
    min-height:100vh;
    overflow-x: hidden;
    background-image: 
        radial-gradient(circle at 50% 50%, #00f2ff05 0%, transparent 50%),
        linear-gradient(rgba(0, 242, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 242, 255, 0.03) 1px, transparent 1px);
    background-size: 100% 100%, 20px 20px, 20px 20px;
  }

  /* Photo Frame Background */
  #photo-frame {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    z-index: -1;
    opacity: 0.15;
    transition: opacity 2s ease-in-out;
    background-size: cover;
    background-position: center;
    filter: grayscale(100%) brightness(50%);
  }

  header { 
    background:rgba(13, 17, 23, 0.9); 
    border-bottom:var(--border); 
    padding:0.75rem 1.5rem; 
    display:flex; 
    align-items:center; 
    gap:1rem;
    box-shadow: var(--glow);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  
  header h1 { font-size:1.2rem; font-weight:700; text-transform: uppercase; letter-spacing: 0.2em; }
  
  .badge { 
    border:var(--border);
    color:var(--text); 
    font-size:.6rem; 
    padding:.2rem .6rem; 
    border-radius:2px; 
    text-transform:uppercase; 
    letter-spacing:.1em;
    box-shadow: var(--glow);
  }
  
  .badge.idle { color:var(--muted); border-color:var(--muted); box-shadow: none; }
  .badge.listening { color:#10b981; border-color:#10b981; box-shadow: 0 0 10px rgba(16, 185, 129, 0.4); }
  .badge.thinking { color:#f59e0b; border-color:#f59e0b; box-shadow: 0 0 10px rgba(245, 158, 11, 0.4); }
  .badge.speaking { color:var(--accent); border-color:var(--accent); animation: pulse 1.5s infinite; }

  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }

  main { 
    max-width:1200px; 
    margin:0 auto; 
    padding:1.5rem; 
    display:grid; 
    grid-template-columns: 1fr 350px; 
    gap:1.5rem; 
  }
  
  @media(max-width:900px){ main{grid-template-columns:1fr} }

  .card { 
    background:rgba(26, 29, 39, 0.7); 
    border-radius:4px; 
    padding:1.25rem; 
    border:var(--border);
    backdrop-filter: blur(8px);
    display: flex;
    flex-direction: column;
    box-shadow: inset 0 0 20px rgba(0, 242, 255, 0.05);
  }
  
  .card h2 { 
    font-size:.7rem; 
    text-transform:uppercase; 
    letter-spacing:.2em; 
    color:var(--accent-dim); 
    margin-bottom:1rem;
    border-bottom: 1px solid #00f2ff11;
    padding-bottom: 0.5rem;
  }

  /* Chat / Terminal */
  #chat { 
    flex: 1;
    display:flex; 
    flex-direction:column; 
    gap:.5rem; 
    max-height:60vh; 
    overflow-y:auto; 
    padding-right:.5rem;
    font-size: 0.85rem;
  }
  
  #chat::-webkit-scrollbar { width: 4px; }
  #chat::-webkit-scrollbar-thumb { background: var(--muted); }

  .msg { margin-bottom: 0.5rem; }
  .msg-header { font-size: 0.65rem; color: var(--muted); margin-bottom: 2px; }
  .msg-content { border-left: 2px solid var(--accent-dim); padding-left: 0.75rem; line-height: 1.4; }
  .msg.user .msg-header { color: var(--accent); }
  .msg.user .msg-content { border-color: var(--accent); }

  .input-row { display:flex; gap:.5rem; margin-top:1.5rem; border-top: 1px solid #00f2ff22; padding-top: 1rem;}
  .input-row input { 
    flex:1; 
    background:transparent; 
    border:none;
    color:var(--text); 
    font-family: inherit;
    font-size:.9rem; 
    outline:none; 
  }
  .input-row span { color: var(--accent); }
  .input-row button { 
    background:transparent; 
    color:var(--accent); 
    border:var(--border); 
    padding:.3rem .8rem; 
    cursor:pointer; 
    font-family: inherit;
    font-size:.7rem;
    text-transform: uppercase;
  }
  .input-row button:hover { background: var(--accent); color: var(--bg); }

  /* Stats and Visualizers */
  .stat { display:flex; justify-content:space-between; align-items:center; padding:.4rem 0; font-size: 0.8rem;}
  .stat-label { color:var(--muted); text-transform: uppercase; font-size: 0.7rem; }
  
  .visualizer-container {
    height: 40px;
    display: flex;
    align-items: flex-end;
    gap: 2px;
    margin-bottom: 1rem;
    padding: 0 5px;
  }
  .v-bar {
    flex: 1;
    background: var(--accent);
    min-height: 2px;
    transition: height 0.1s ease;
    opacity: 0.6;
  }

  /* Interactive Controls */
  .control-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }
  .btn-ctrl {
    background: transparent;
    border: var(--border);
    color: var(--text);
    padding: 0.5rem;
    font-size: 0.65rem;
    text-transform: uppercase;
    cursor: pointer;
    text-align: center;
    transition: all 0.2s;
  }
  .btn-ctrl:hover {
    background: rgba(0, 242, 255, 0.1);
    box-shadow: var(--glow);
  }
  .btn-ctrl.active {
    background: var(--accent);
    color: var(--bg);
  }

  .empty { color:var(--muted); font-size:.75rem; text-align:center; padding:2rem 0; font-style: italic; }

</style>
</head>
<body>
<div id="photo-frame"></div>
<header>
  <h1>ARIA // CORE_INTERFACE</h1>
  <span class="badge idle" id="state-badge">idle</span>
</header>
<main>
  <div>
    <div class="card">
      <h2>Active Logs</h2>
      <div id="chat"><p class="empty">SYSTEM READY. WAITING FOR INPUT...</p></div>
      <div class="input-row">
        <span>></span>
        <input id="txt" type="text" placeholder="ENTER COMMAND..." autocomplete="off">
        <button onclick="sendMsg()">Execute</button>
      </div>
    </div>
  </div>
  
  <div style="display:flex;flex-direction:column;gap:1rem">
    <div class="card">
        <h2>Audio Spectrum</h2>
        <div class="visualizer-container" id="visualizer">
            <!-- Bars added by JS -->
        </div>
        <div id="stats"></div>
    </div>
    
    <div class="card">
      <h2>Media & Hardware</h2>
      <div class="control-grid">
          <button class="btn-ctrl" onclick="cmd('music shuffle')">Shuffle Music</button>
          <button class="btn-ctrl" onclick="cmd('stop music')">Stop Music</button>
          <button class="btn-ctrl" onclick="cmd('start spotify')">Spotify On</button>
          <button class="btn-ctrl" onclick="cmd('stop spotify')">Spotify Off</button>
          <button class="btn-ctrl" onclick="cmd('radio dlf')">Radio DLF</button>
          <button class="btn-ctrl" onclick="cmd('radio jazz')">Radio Jazz</button>
          <button class="btn-ctrl" onclick="cmd('turn on lights')">Lights On</button>
          <button class="btn-ctrl" onclick="cmd('turn off lights')">Lights Off</button>
      </div>
      <div id="tools" style="margin-top: 1rem; border-top: 1px solid #00f2ff11; padding-top: 0.5rem;"></div>
    </div>
  </div>
</main>

<script>
let lastLen = 0;
const vizBars = 24;
const vizContainer = document.getElementById('visualizer');
for(let i=0; i<vizBars; i++) {
    const bar = document.createElement('div');
    bar.className = 'v-bar';
    vizContainer.appendChild(bar);
}

function updateVisualizer(state) {
    const bars = document.querySelectorAll('.v-bar');
    bars.forEach(bar => {
        let h = 2;
        if(state === 'speaking' || state === 'listening' || state === 'listening:wake') {
            h = Math.floor(Math.random() * 38) + 2;
        } else if (state === 'thinking') {
            h = 10 + Math.sin(Date.now()/200) * 8;
        }
        bar.style.height = h + 'px';
    });
}

async function updatePhoto() {
    try {
        const r = await fetch('/api/photo');
        const d = await r.json();
        if(d.url) {
            const frame = document.getElementById('photo-frame');
            frame.style.backgroundImage = `url(${d.url})`;
        }
    } catch(e) {}
}

async function poll() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();

    // Badge
    const b = document.getElementById('state-badge');
    b.textContent = d.state;
    b.className = 'badge ' + d.state.replace(':', '-');

    updateVisualizer(d.state);

    // Chat
    if (d.history.length !== lastLen) {
      lastLen = d.history.length;
      const el = document.getElementById('chat');
      if (d.history.length === 0) {
        el.innerHTML = '<p class="empty">SYSTEM IDLE.</p>';
      } else {
        el.innerHTML = d.history.map(m => `
          <div class="msg ${m.role}">
            <div class="msg-header">[${m.ts}] ${m.role.toUpperCase()}</div>
            <div class="msg-content">${m.text}</div>
          </div>`).join('');
        el.scrollTop = el.scrollHeight;
      }
    }

    // Stats
    document.getElementById('stats').innerHTML = d.system.map(s =>
      `<div class="stat"><span class="stat-label">${s.label}</span><span class="stat-value">${s.value}</span></div>`
    ).join('');

    // Tools
    document.getElementById('tools').innerHTML = d.tools.map(t => 
        `<div class="stat"><span class="stat-label">${t.name}</span><span class="stat-value">${t.status}</span></div>`
    ).join('');

  } catch(e) {}
  setTimeout(poll, 800);
}

async function sendMsg() {
  const inp = document.getElementById('txt');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  await cmd(text);
}

async function cmd(text) {
    await fetch('/api/send', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
}

document.getElementById('txt').addEventListener('keydown', e => { if(e.key==='Enter') sendMsg(); });

// Initial photo and periodic refresh
updatePhoto();
setInterval(updatePhoto, 30000);
poll();
</script>
</body>
</html>"""

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/api/state")
    def api_state():
        import psutil
        system_stats = []
        try:
            # CPU and RAM are fast
            system_stats = [
                {"label": "CPU_LOAD", "value": f"{psutil.cpu_percent(interval=None):.1f}%"},
                {"label": "MEM_USED", "value": f"{psutil.virtual_memory().percent:.1f}%"},
                {"label": "THERMAL", "value": _get_cpu_temp()},
                {"label": "UPTIME", "value": _get_uptime()},
            ]
        except Exception:
            pass

        tool_status = []
        for name, tool in _tools_ref.items():
            if hasattr(tool, "get_status"):
                try:
                    tool_status.append({"name": name, "status": tool.get_status()})
                except Exception:
                    pass

        with _history_lock:
            history_snapshot = list(_history)

        return jsonify({
            "history": history_snapshot,
            "state": _status["state"],
            "system": system_stats,
            "tools": tool_status,
        })

    @app.route("/api/send", methods=["POST"])
    def api_send():
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if text:
            _command_queue.put(text)
        return jsonify({"ok": True})

    @app.route("/api/photo")
    def api_photo():
        if not _photo_dir or not _photo_dir.exists():
            return jsonify({"url": None})
        
        photos = []
        for ext in ('.jpg', '.jpeg', '.png', '.webp'):
            photos.extend(list(_photo_dir.glob(f"*{ext}")))
            photos.extend(list(_photo_dir.glob(f"*{ext.upper()}")))
        
        if not photos:
            return jsonify({"url": None})
        
        # Pick a random photo
        photo = random.choice(photos)
        return jsonify({"url": f"/photos/{photo.name}"})

    @app.route("/photos/<path:filename>")
    def get_photo(filename):
        if not _photo_dir:
            return "Not found", 404
        return send_from_directory(_photo_dir, filename)

    return app


def _get_cpu_temp() -> str:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return f"{int(f.read().strip()) / 1000:.1f} °C"
    except OSError:
        return "N/A"


def _get_uptime() -> str:
    try:
        import psutil
        boot = psutil.boot_time()
        secs = int(time.time() - boot)
        h, m = divmod(secs // 60, 60)
        return f"{h}h {m}m"
    except Exception:
        return "N/A"


class Dashboard:
    def __init__(self, config: dict, tools: dict):
        self.cfg = config.get("dashboard", {})
        global _tools_ref, _photo_dir
        _tools_ref = tools
        
        # Determine photo directory
        project_root = Path(__file__).parent.parent
        self.photo_dir = project_root / "media" / "photos"
        _photo_dir = self.photo_dir
        
        self._thread: Optional[threading.Thread] = None
        self._app = None

    def start(self):
        if not self.cfg.get("enabled", True):
            return
        self._app = _build_app()
        if self._app is None:
            logger.warning("Dashboard: Flask not installed — dashboard disabled.")
            return

        host = self.cfg.get("host", "0.0.0.0")
        port = self.cfg.get("port", 5000)

        def _run():
            import logging as _logging
            _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
            self._app.run(host=host, port=port, use_reloader=False, threaded=True)

        self._thread = threading.Thread(target=_run, daemon=True, name="dashboard")
        self._thread.start()
        logger.info("Dashboard running at http://0.0.0.0:%d", port)

    # Proxy push functions so main.py doesn't import dashboard internals
    @staticmethod
    def push_user(text: str): push_user(text)

    @staticmethod
    def push_assistant(text: str): push_assistant(text)

    @staticmethod
    def set_state(state: str): set_state(state)

    @staticmethod
    def get_next_command(timeout: float = 0.1) -> Optional[str]:
        return get_next_command(timeout)
