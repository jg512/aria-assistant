#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — One-shot installer for Aria on Raspberry Pi 4
# Tested on: Raspberry Pi OS Bookworm (64-bit) — 4 GB or 8 GB RAM
#
# Usage (run as the pi user, NOT as root):
#   git clone https://github.com/yourname/aria ~/aria
#   cd ~/aria && chmod +x setup.sh && ./setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
error()   { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }
section() { echo -e "\n${CYAN}━━━ $* ━━━${NC}"; }

# ── Sanity checks ─────────────────────────────────────────────────────────────
[[ "$EUID" -eq 0 ]] && error "Do NOT run as root. Run as your normal user (e.g. pi)."
command -v apt-get &>/dev/null || error "This script requires a Debian/Ubuntu system."

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HOME/models"
DATA_DIR="$INSTALL_DIR/data"
LOG_DIR="$INSTALL_DIR/logs"
PIPER_VERSION="1.2.0"
PIPER_VOICE="en_US-lessac-medium"
PIPER_ARCH="aarch64"

# Detect architecture
MACHINE=$(uname -m)
if [[ "$MACHINE" == "aarch64" || "$MACHINE" == "arm64" ]]; then
    PIPER_ARCH="aarch64"
elif [[ "$MACHINE" == "armv7l" ]]; then
    PIPER_ARCH="armv7l"
elif [[ "$MACHINE" == "x86_64" ]]; then
    PIPER_ARCH="x86_64"
    warn "Running on x86_64 — Piper and models will use the x86_64 build."
    warn "For best performance, deploy on Raspberry Pi 4 (aarch64)."
else
    warn "Unknown architecture '$MACHINE' — defaulting to x86_64 for Piper."
    PIPER_ARCH="x86_64"
fi

info "Installing Aria to: $INSTALL_DIR"
info "Architecture: $PIPER_ARCH"

# ── System packages ───────────────────────────────────────────────────────────
section "System packages"
sudo apt-get update -qq
sudo apt-get install -y \
    git curl wget python3 python3-pip python3-venv \
    portaudio19-dev libportaudio2 libasound2-dev \
    mpv ffmpeg alsa-utils \
    build-essential cmake libatlas-base-dev

info "System packages installed."

# ── Ollama ────────────────────────────────────────────────────────────────────
section "Ollama (LLM runtime)"
if ! command -v ollama &>/dev/null; then
    info "Installing Ollama …"
    curl -fsSL https://ollama.com/install.sh | sh
else
    info "Ollama already installed: $(ollama --version)"
fi

# Start Ollama — try systemd first, fall back to background process
mkdir -p "$LOG_DIR"
if systemctl is-active --quiet ollama 2>/dev/null; then
    info "Ollama service already running."
elif sudo systemctl enable ollama 2>/dev/null && sudo systemctl start ollama 2>/dev/null; then
    info "Ollama started via systemd."
    sleep 3
else
    warn "systemd unavailable — starting Ollama as background process ..."
    nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    OLLAMA_PID=$!
    sleep 5
    if kill -0 "$OLLAMA_PID" 2>/dev/null; then
        info "Ollama running in background (PID $OLLAMA_PID)."
    else
        warn "Ollama failed to start. Open a second terminal and run: ollama serve"
        warn "Then re-run: bash setup.sh"
    fi
fi

info "Pulling TinyLlama model (may take a few minutes) ..."
for attempt in 1 2 3; do
    ollama pull tinyllama && break
    warn "Pull attempt $attempt failed — retrying in 5 s ..."
    sleep 5
done || warn "Could not pull tinyllama. Run manually: ollama pull tinyllama"

# ── Python virtual environment ────────────────────────────────────────────────
section "Python environment"
if [ ! -d "$INSTALL_DIR/venv" ]; then
    python3 -m venv "$INSTALL_DIR/venv"
fi
source "$INSTALL_DIR/venv/bin/activate"
pip install --upgrade pip --quiet

info "Installing Python packages ..."

# Install core packages (excluding openwakeword — handled separately below)
pip install --quiet \
    faster-whisper \
    pyaudio \
    numpy \
    requests \
    flask \
    psutil \
    pytest \
    pytest-timeout

# openwakeword: tflite-runtime does not exist for Python 3.12+ on x86_64.
# Install with --no-deps to skip it, then add onnxruntime (fully supported everywhere).
# openwakeword supports both tflite and onnxruntime — onnxruntime is preferred.
info "Installing openwakeword with onnxruntime backend ..."
pip install --quiet --no-deps openwakeword
pip install --quiet "onnxruntime>=1.16" soundfile

info "Python packages installed."

# ── Piper TTS binary ──────────────────────────────────────────────────────────
section "Piper TTS binary"
mkdir -p "$MODEL_DIR"

PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_${PIPER_ARCH}.tar.gz"

if [ ! -f /usr/local/bin/piper ]; then
    info "Downloading Piper ${PIPER_VERSION} for ${PIPER_ARCH} …"
    TMP=$(mktemp -d)
    if wget -q --show-progress "$PIPER_URL" -O "$TMP/piper.tar.gz"; then
        tar -xzf "$TMP/piper.tar.gz" -C "$TMP"
        sudo cp "$TMP/piper/piper" /usr/local/bin/piper
        sudo chmod +x /usr/local/bin/piper
        info "Piper installed at /usr/local/bin/piper"
    else
        warn "Could not download Piper binary. Install manually from:"
        warn "  https://github.com/rhasspy/piper/releases"
    fi
    rm -rf "$TMP"
else
    info "Piper already installed."
fi

# ── Piper voice model ─────────────────────────────────────────────────────────
section "Piper voice model (${PIPER_VOICE})"
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

if [ ! -f "$MODEL_DIR/${PIPER_VOICE}.onnx" ]; then
    info "Downloading voice model …"
    wget -q --show-progress "${VOICE_BASE}/${PIPER_VOICE}.onnx"      -O "$MODEL_DIR/${PIPER_VOICE}.onnx"
    wget -q              "${VOICE_BASE}/${PIPER_VOICE}.onnx.json" -O "$MODEL_DIR/${PIPER_VOICE}.onnx.json"
    info "Voice model saved to $MODEL_DIR/"
else
    info "Voice model already present."
fi

# Update config.json with actual paths
ESCAPED_MODEL_DIR="${MODEL_DIR//\//\\/}"
sed -i "s|/home/pi/models|${MODEL_DIR}|g" "$INSTALL_DIR/config.json" 2>/dev/null || true

# ── OpenWakeWord models ───────────────────────────────────────────────────────
section "OpenWakeWord models"
python3 - <<'EOF'
try:
    import openwakeword
    openwakeword.utils.download_models()
    print("OpenWakeWord models downloaded.")
except Exception as e:
    print(f"Warning: could not pre-download OWW models: {e}")
    print("They will be downloaded on first run.")
EOF

# ── Data and log directories ──────────────────────────────────────────────────
section "Data directories"
mkdir -p "$DATA_DIR" "$LOG_DIR" "$HOME/Music"
# Update config.json paths for this user
sed -i "s|/home/pi|${HOME}|g" "$INSTALL_DIR/config.json" 2>/dev/null || true
info "Directories ready: $DATA_DIR, $LOG_DIR, $HOME/Music"

# ── Detect USB audio device ───────────────────────────────────────────────────
section "USB audio device detection"
USB_CARD=$(aplay -l 2>/dev/null | grep -i "usb\|USB" | head -1 | grep -oP 'card \K[0-9]+' || echo "")
if [ -n "$USB_CARD" ]; then
    USB_DEV="plughw:${USB_CARD},0"
    sed -i "s|plughw:CARD=Speaker,DEV=0|${USB_DEV}|g" "$INSTALL_DIR/config.json" 2>/dev/null || true
    info "USB audio device detected: card $USB_CARD → $USB_DEV (set in config.json)"
else
    warn "No USB audio device found. Edit 'tts.output_device' in config.json manually."
    warn "Run 'aplay -l' to list available devices."
fi

# ── Run tests ─────────────────────────────────────────────────────────────────
section "Running tests"
if python3 -m pytest "$INSTALL_DIR/tests/" -q --timeout=10 2>&1; then
    info "All tests passed."
else
    warn "Some tests failed — this is OK if hardware libraries aren't installed yet."
fi

# ── systemd service ───────────────────────────────────────────────────────────
section "systemd service (auto-start on boot)"
SERVICE_FILE="/etc/systemd/system/aria.service"
sudo tee "$SERVICE_FILE" > /dev/null <<SERVICE
[Unit]
Description=Aria — Local AI Companion
After=network.target ollama.service sound.target
Wants=ollama.service

[Service]
Type=simple
User=${USER}
WorkingDirectory=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/venv/bin/python3 ${INSTALL_DIR}/main.py
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable aria
info "systemd service installed."

# ── sudo permissions for shutdown/reboot ──────────────────────────────────────
section "Sudo permissions"
SUDOERS_LINE="${USER} ALL=(ALL) NOPASSWD: /sbin/shutdown, /sbin/reboot"
SUDOERS_FILE="/etc/sudoers.d/aria"
echo "$SUDOERS_LINE" | sudo tee "$SUDOERS_FILE" > /dev/null
sudo chmod 440 "$SUDOERS_FILE"
info "Passwordless shutdown/reboot granted."

# ── Done ──────────────────────────────────────────────────────────────────────
section "Installation complete"
echo -e "${GREEN}"
cat << 'BANNER'
    _         _
   / \   _ __(_) __ _
  / _ \ | '__| |/ _` |
 / ___ \| |  | | (_| |
/_/   \_\_|  |_|\__,_|

  Local AI Companion — Ready
BANNER
echo -e "${NC}"
echo "  Next steps:"
echo "    1. Add music:          cp *.mp3 ~/Music/"
echo "    2. Configure weather:  edit config.json → weather.latitude/longitude/location_name"
echo "    3. Home Assistant:     edit config.json → home_assistant.enabled + token"
echo "    4. Export calendar:    save .ics to ~/calendar.ics"
echo ""
echo "  Start:"
echo "    sudo systemctl start aria"
echo "    journalctl -u aria -f"
echo ""
echo "  Test without wake word:"
echo "    source venv/bin/activate && python3 main.py --no-wake-word"
echo ""
echo "  Dashboard:   http://$(hostname).local:5000"
echo "  Hot reload:  kill -HUP \$(pgrep -f main.py)"
echo ""
echo -e "  Say ${YELLOW}'Hey Aria'${NC} to wake her up!"