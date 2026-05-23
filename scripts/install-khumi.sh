#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Project Khumi — Install Script
# Run on a Raspberry Pi 5 with IIAB already installed.
# Usage: sudo bash install-khumi.sh
# ============================================================

KHUMI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_DIR="/home/${SUDO_USER:-pi}/iiab-agent"
LLAMA_DIR="/home/${SUDO_USER:-pi}/llama.cpp"
MODEL_DIR="$AGENT_DIR/models"
USER="${SUDO_USER:-pi}"

echo "=== Project Khumi Installer ==="
echo "User: $USER"
echo "Agent dir: $AGENT_DIR"
echo ""

# --- 1. System dependencies ---
echo "[1/8] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq build-essential cmake libopenblas-dev \
    python3-venv python3-pip git wget curl ffmpeg

# --- 2. Python venv + packages ---
echo "[2/8] Setting up Python environment..."
if [ ! -d "$AGENT_DIR/bin" ]; then
    sudo -u "$USER" python3 -m venv "$AGENT_DIR"
fi
sudo -u "$USER" "$AGENT_DIR/bin/pip" install --quiet \
    fastapi uvicorn httpx python-multipart piper-tts faster-whisper

# --- 3. Build llama.cpp ---
echo "[3/8] Building llama.cpp from source..."
if [ ! -f "$LLAMA_DIR/build/bin/llama-server" ]; then
    if [ ! -d "$LLAMA_DIR" ]; then
        sudo -u "$USER" git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
    fi
    sudo -u "$USER" cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON
    sudo -u "$USER" cmake --build "$LLAMA_DIR/build" --config Release -j4
fi
echo "  llama-server: $LLAMA_DIR/build/bin/llama-server"

# --- 4. Download models ---
echo "[4/8] Downloading LLM and voice models..."
sudo -u "$USER" mkdir -p "$MODEL_DIR"

# LLM: qwen2.5 0.5B (fast, ~470MB)
if [ ! -f "$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf" ]; then
    sudo -u "$USER" wget -q --show-progress -O "$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
fi

# LLM: qwen2.5 1.5B (quality, ~1GB)
if [ ! -f "$MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf" ]; then
    sudo -u "$USER" wget -q --show-progress -O "$MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
fi

# Piper TTS voice
if [ ! -f "$MODEL_DIR/en_US-lessac-medium.onnx" ]; then
    sudo -u "$USER" wget -q --show-progress -O "$MODEL_DIR/en_US-lessac-medium.onnx" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    sudo -u "$USER" wget -q --show-progress -O "$MODEL_DIR/en_US-lessac-medium.onnx.json" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
fi

# --- 5. Install app ---
echo "[5/8] Installing Khumi agent app..."
sudo -u "$USER" cp "$KHUMI_DIR/agent/app.py" "$AGENT_DIR/app.py"

# --- 6. Install systemd services ---
echo "[6/8] Installing systemd services..."

# LLM service
sed "s|/home/drichards13|/home/$USER|g" "$KHUMI_DIR/systemd/khumi-llm.service" \
    | sed "s|User=drichards13|User=$USER|g" \
    > /etc/systemd/system/khumi-llm.service

# Agent service (adapt from existing or create)
if [ -f "$KHUMI_DIR/systemd/iiab-agent.service" ]; then
    sed "s|/home/drichards13|/home/$USER|g" "$KHUMI_DIR/systemd/iiab-agent.service" \
        | sed "s|User=drichards13|User=$USER|g" \
        > /etc/systemd/system/iiab-agent.service
fi

systemctl daemon-reload
systemctl enable khumi-llm.service
systemctl enable iiab-agent.service

# --- 7. Set CPU governor ---
echo "[7/8] Setting CPU to performance mode..."
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

# --- 8. Start services ---
echo "[8/8] Starting services..."
systemctl restart khumi-llm.service
sleep 5
systemctl restart iiab-agent.service
sleep 5

# Verify
echo ""
echo "=== Khumi Install Complete ==="
LLM_OK=$(curl -s --max-time 5 http://127.0.0.1:8070/health 2>/dev/null | grep -c ok || true)
AGENT_OK=$(curl -s --max-time 5 http://localhost:8090/api/status 2>/dev/null | grep -c ok || true)
echo "LLM server:  $([ "$LLM_OK" -gt 0 ] && echo '✓ running' || echo '✗ not responding')"
echo "Khumi agent: $([ "$AGENT_OK" -gt 0 ] && echo '✓ running' || echo '✗ not responding')"
echo ""
echo "Open http://$(hostname -I | awk '{print $1}')/agent/ in a browser."
echo ""
