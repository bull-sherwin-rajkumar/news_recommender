#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "Setup failed at line $LINENO." >&2' ERR

die() {
    echo "Error: $*" >&2
    exit 1
}

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[[ -f pyproject.toml && -f app.py && -f .env.example ]] ||
    die "Run this script from the cloned project."

[[ "$ROOT" =~ ^[a-zA-Z0-9_./-]+$ ]] ||
    die "Repository path must contain only letters, numbers, _, ., /, or -."

[[ "$EUID" -ne 0 ]] ||
    die "Run as your regular EC2 user, not root."

sudo -n true ||
    die "Passwordless sudo is required."

source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] ||
    die "This script supports Ubuntu 24.04."

read -r -p "Ollama model (for example qwen3.5:4b): " MODEL
[[ "$MODEL" =~ ^[a-zA-Z0-9][a-zA-Z0-9._:/-]*$ ]] ||
    die "Enter a valid Ollama model name."

export DEBIAN_FRONTEND=noninteractive
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl zstd build-essential \
    python3.12 python3.12-venv python3-dev

# Stop the managed frontend before changing dependencies or indexes.
if sudo -n systemctl cat news-explorer.service >/dev/null 2>&1; then
    sudo -n systemctl stop news-explorer.service
fi

if ! command -v ollama >/dev/null 2>&1; then
    INSTALLER="$(mktemp)"
    trap 'rm -f -- "$INSTALLER"' EXIT
    curl --fail --silent --show-error --location \
        https://ollama.com/install.sh -o "$INSTALLER"
    sh "$INSTALLER"
    rm -f -- "$INSTALLER"
    trap - EXIT
fi

# Bind Ollama locally even if the host environment has other defaults.
sudo -n mkdir -p /etc/systemd/system/ollama.service.d
sudo -n tee /etc/systemd/system/ollama.service.d/news-local.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
EOF

sudo -n systemctl daemon-reload
sudo -n systemctl enable ollama
sudo -n systemctl restart ollama

OLLAMA_READY=false
for ((attempt = 0; attempt < 60; attempt++)); do
    if curl --fail --silent --max-time 2 \
        http://127.0.0.1:11434/api/tags >/dev/null; then
        OLLAMA_READY=true
        break
    fi
    sleep 2
done

[[ "$OLLAMA_READY" == true ]] ||
    die "Ollama did not start. Check: sudo journalctl -u ollama"

export OLLAMA_HOST=http://127.0.0.1:11434
ollama pull "$MODEL"

if [[ ! -f .env ]]; then
    cp .env.example .env
    sed -i -E '/^[[:space:]]*(export[[:space:]]+)?NEWS_DEVICE[[:space:]]*=/d' .env
    printf '\nNEWS_DEVICE=auto\n' >> .env
fi

chmod 600 .env

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

# Update only local Ollama settings; preserve embedding/reranking settings.
.venv/bin/python - "$MODEL" <<'PY'
import sys
from dotenv import set_key

set_key(".env", "OLLAMA_MODEL", sys.argv[1])
set_key(".env", "OLLAMA_HOST", "http://127.0.0.1:11434")
PY

echo "Downloading and warming embedding/reranking models..."
.venv/bin/python <<'PY'
from news_recommender.config import Settings
from news_recommender.nlp import NLP

settings = Settings.from_env()
print(f"Embedding: {settings.embedding_model}")
print(f"Reranker: {settings.reranker_model}")
print(f"Device: {settings.device}")
NLP(settings).warmup()
PY

echo "Indexing the full dataset; this can take considerable time..."
.venv/bin/news-ingest

echo "Checking retrieval and Ollama initialization..."
.venv/bin/python <<'PY'
import sys
from dataclasses import replace

from dotenv import set_key

from news_recommender.chat import Chat
from news_recommender.config import Settings
from news_recommender.services import Services

settings = Settings.from_env()
services = Services(settings)
retriever = services.retrieval.result()
chat = services.chat.result()
if chat.warmup_warning:
    if settings.context_tokens != 64000:
        raise RuntimeError(chat.warmup_warning)
    print(
        f"{chat.warmup_warning}\nRetrying Ollama with an 8,192-token context...",
        file=sys.stderr,
    )
    Chat(replace(settings, context_tokens=8192), retriever).warmup()
    set_key(".env", "NEWS_CONTEXT_TOKENS", "8192")
    print("Ollama initialized successfully; saved NEWS_CONTEXT_TOKENS=8192 in .env.")
PY

APP_USER="$(id -un)"
APP_GROUP="$(id -gn)"

sudo -n tee /etc/systemd/system/news-explorer.service >/dev/null <<EOF
[Unit]
Description=News Explorer
Wants=network-online.target
Requires=ollama.service
After=network-online.target ollama.service

[Service]
Type=simple
User=$APP_USER
Group=$APP_GROUP
WorkingDirectory=$ROOT
Environment="HOME=$HOME"
Environment="PYTHONUNBUFFERED=1"
ExecStart=$ROOT/.venv/bin/python -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
Restart=on-failure
RestartSec=5
UMask=0077

[Install]
WantedBy=multi-user.target
EOF

sudo -n systemctl daemon-reload
sudo -n systemctl enable news-explorer.service
sudo -n systemctl restart news-explorer.service

for ((attempt = 0; attempt < 60; attempt++)); do
    if sudo -n systemctl is-active --quiet news-explorer.service &&
        curl --fail --silent --max-time 2 \
            http://127.0.0.1:8501/_stcore/health >/dev/null; then
        echo "Setup complete. Open http://<EC2-public-IP>:8501"
        echo "Logs: sudo journalctl -u news-explorer -f"
        exit 0
    fi
    sleep 2
done

die "App health check failed. Check: sudo journalctl -u news-explorer"