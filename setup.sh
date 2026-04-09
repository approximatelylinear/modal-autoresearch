#!/usr/bin/env bash
set -euo pipefail

# modal-autoresearch setup
# Usage: ./setup.sh [--project PATH_TO_PROJECT]
#
# Installs dependencies, authenticates with Modal, and optionally
# links a project for autoresearch.

BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${BOLD}$1${NC}"; }
ok()    { echo -e "  ${GREEN}✓${NC} $1"; }
warn()  { echo -e "  ${YELLOW}!${NC} $1"; }
fail()  { echo -e "  ${RED}✗${NC} $1"; }
step()  { echo -e "\n${BOLD}[$1/$TOTAL_STEPS] $2${NC}"; }

TOTAL_STEPS=5
PROJECT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --project) PROJECT_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./setup.sh [--project PATH_TO_PROJECT]"
            echo ""
            echo "Sets up modal-autoresearch: installs deps, authenticates"
            echo "with Modal, and configures your OpenAI API key."
            echo ""
            echo "Options:"
            echo "  --project PATH  Path to a project with autoresearch.toml"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

info "modal-autoresearch setup"
echo ""

# ---- 1. Check for uv ----
step 1 "Checking for uv"
if command -v uv &> /dev/null; then
    ok "uv $(uv --version 2>&1 | head -1)"
else
    warn "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &> /dev/null; then
        ok "uv installed: $(uv --version 2>&1 | head -1)"
    else
        fail "Could not install uv. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi

# ---- 2. Install Python dependencies ----
step 2 "Installing dependencies"
uv sync --quiet
ok "Dependencies installed"

# ---- 3. Modal authentication ----
step 3 "Modal authentication"
if uv run python -c "import modal; modal.Client.from_credentials()" &> /dev/null; then
    ok "Already authenticated with Modal"
else
    warn "Not authenticated with Modal"
    echo ""
    echo "    Run this command to authenticate:"
    echo "      uv run modal token new"
    echo ""
    read -rp "    Authenticate now? [Y/n] " auth_now
    if [[ "${auth_now:-y}" =~ ^[Yy] ]]; then
        uv run modal token new || {
            fail "Modal authentication failed"
            echo "    Try manually: uv run modal token new"
        }
    else
        warn "Skipped — run 'uv run modal token new' before launching experiments"
    fi
fi

# ---- 4. OpenAI API key ----
step 4 "OpenAI API key"
if [ -f .env ] && grep -q "OPENAI_API_KEY" .env 2>/dev/null; then
    ok ".env exists with OPENAI_API_KEY"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
    ok "OPENAI_API_KEY set in environment"
else
    warn "No OpenAI API key found"
    echo ""
    read -rp "    Enter your OpenAI API key (or press Enter to skip): " api_key
    if [ -n "$api_key" ]; then
        echo "OPENAI_API_KEY=$api_key" >> .env
        ok "Saved to .env"
    else
        warn "Skipped — create .env with OPENAI_API_KEY=sk-... before running the agent"
    fi
fi

# ---- 5. Project validation ----
step 5 "Project check"
if [ -n "$PROJECT_PATH" ]; then
    PROJECT_PATH=$(realpath "$PROJECT_PATH")
    if [ -f "$PROJECT_PATH/autoresearch.toml" ]; then
        ok "Found autoresearch.toml in $PROJECT_PATH"
        # Validate manifest loads
        if uv run python -c "
from autoresearch.manifest import load_manifest
m = load_manifest('$PROJECT_PATH/autoresearch.toml')
print(f'  Project: {m.name}')
print(f'  Phases: {list(m.phases.keys())}')
print(f'  Primary metric: {m.metrics.primary}')
" 2>&1; then
            ok "Manifest validated"
        else
            fail "Manifest validation failed — check autoresearch.toml"
        fi
    else
        fail "No autoresearch.toml found in $PROJECT_PATH"
        echo "    See README.md for how to make a project autoresearchable"
    fi
else
    warn "No project specified (use --project PATH to validate one)"
fi

# ---- Done ----
echo ""
info "Setup complete!"
echo ""
echo "  Quick start:"
echo "    ${DIM}# Interactive session${NC}"
echo "    uv run python run_session.py"
echo ""
echo "    ${DIM}# LLM agent with approval gates${NC}"
echo "    uv run python run_agent.py --hitl --import-tsv path/to/results.tsv"
echo ""
echo "    ${DIM}# Fully autonomous${NC}"
echo "    uv run python run_agent.py --max-turns 30"
echo ""
echo "    ${DIM}# Validate substrate (run first!)${NC}"
echo "    uv run modal run smoke_baseline.py"
echo ""
