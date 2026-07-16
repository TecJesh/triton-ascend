#!/usr/bin/env bash
# ============================================================================
# sync-upstream.sh — Runnable shell equivalent of .github/workflows/sync-upstream.yml
#
# Usage:
#   TARGET_COMMIT=<sha> ./tmp/sync-upstream.sh    # sync to a specific upstream commit
#   ./tmp/sync-upstream.sh                         # sync to latest upstream commit
#
# Required environment variables (set before running):
#   ANTHROPIC_API_KEY   — DeepSeek API key (used via Anthropic-compatible API)
#   SYNC_PAT            — PAT with `repo` + `workflow` scope for push + PR
#                         (falls back to GITHUB_TOKEN, then gh auth status)
#
# Optional:
#   TARGET_COMMIT       — Upstream triton commit SHA (empty = latest)
#   GITHUB_TOKEN        — Fallback token if SYNC_PAT is not set
#   SKIP_BISHENG        — Set to 1 to skip Bisheng compiler download/install
#   SKIP_CLAUDE         — Set to 1 to skip Claude Code CLI install
#   SKIP_LLVM           — Set to 1 to skip llvm-project clone
# ============================================================================

set -euo pipefail

# ── Configurable paths ─────────────────────────────────────────────────────
TRITON_ASCEND_PATH="${TRITON_ASCEND_PATH:-$(cd "$(dirname "$0")/.." && pwd)}"
TRITON_PATH="${TRITON_PATH:-$HOME/triton-upstream}"
TA_MAIN2MAIN_WORKSPACE="${TA_MAIN2MAIN_WORKSPACE:-/tmp/ta-workspace}"
LLVM_PROJECT_PATH="${LLVM_PROJECT_PATH:-$HOME/llvm-project}"
LLVM_INSTALL_PREFIX_SYNC="${LLVM_INSTALL_PREFIX_SYNC:-$HOME/llvm-install-sync}"

# ── Environment defaults (matching the workflow) ────────────────────────────
PYTHON="${PYTHON:-python3.11}"
export PROTON_SKIP_PC_SAMPLING_TEST=1
# export PIP_INDEX_URL="${PIP_INDEX_URL:-http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple}"
# export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-cache-service.nginx-pypi-cache.svc.cluster.local}"
# export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://repo.huaweicloud.com/ascend/repos/pypi}"

# ── Helper functions ────────────────────────────────────────────────────────
log()  { echo "==> $(date '+%H:%M:%S')  $*"; }
warn() { echo "WARN $(date '+%H:%M:%S')  $*" >&2; }
die()  { echo "ERROR $(date '+%H:%M:%S')  $*" >&2; exit 1; }

# ── Pre-flight checks ──────────────────────────────────────────────────────
log "Working directory: $TRITON_ASCEND_PATH"
cd "$TRITON_ASCEND_PATH"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  die "ANTHROPIC_API_KEY is not set. Export it before running this script."
fi

GH_TOKEN="${SYNC_PAT:-${GITHUB_TOKEN:-}}"
if [ -z "${GH_TOKEN:-}" ]; then
  if command -v gh &>/dev/null && gh auth status &>/dev/null 2>&1; then
    warn "SYNC_PAT / GITHUB_TOKEN not set; falling back to 'gh auth token'"
    GH_TOKEN=$(gh auth token 2>/dev/null || true)
  fi
  if [ -z "${GH_TOKEN:-}" ]; then
    die "SYNC_PAT, GITHUB_TOKEN, or a valid 'gh auth' session is required."
  fi
fi
export GH_TOKEN

# ── Step 1: Git safe directory + identity ───────────────────────────────────
log "Configuring git safe directory + identity"
git config --global --add safe.directory "$TRITON_ASCEND_PATH"
git config user.name  "TA Sync Bot"
git config user.email "ta-sync-bot@users.noreply.github.com"

# ── Step 2: Install GitHub CLI ──────────────────────────────────────────────
log "Checking GitHub CLI"
if ! command -v gh &>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq gh
fi
log "gh: $(gh --version | head -1)"

# ── Step 3: Add upstream triton remote ──────────────────────────────────────
log "Configuring upstream-triton remote"
if ! git remote get-url upstream-triton 2>/dev/null; then
  git remote add upstream-triton https://github.com/triton-lang/triton.git
fi
git fetch upstream-triton --prune --tags

# ── Step 4: Clone upstream triton (bare) ────────────────────────────────────
log "Preparing upstream triton bare clone at $TRITON_PATH"
if [ ! -d "$TRITON_PATH/.git" ] && [ ! -f "$TRITON_PATH/HEAD" ]; then
  git clone --bare https://github.com/triton-lang/triton.git "$TRITON_PATH"
else
  git -C "$TRITON_PATH" fetch origin --prune --tags
fi
log "upstream triton ready at $TRITON_PATH ($(git -C "$TRITON_PATH" rev-parse --short HEAD))"

# # ── Step 5: Download and install Bisheng compiler ───────────────────────────
# if [ "${SKIP_BISHENG:-0}" != "1" ]; then
#   log "Installing Bisheng compiler"
#   if [ ! -f cmake/bisheng-version.txt ]; then
#     warn "cmake/bisheng-version.txt not found, skipping Bisheng install"
#   else
#     OBS_BASE="https://triton-ascend-artifacts.obs.cn-southwest-2.myhuaweicloud.com"
#     BISHENG_FILENAME=$(sed -n '1p' cmake/bisheng-version.txt | xargs)
#     BISHENG_SHA256=$(sed -n '2p' cmake/bisheng-version.txt | sed 's/^sha256://' | xargs)

#     if [ -z "${BISHENG_FILENAME}" ]; then
#       die "cmake/bisheng-version.txt line 1 must contain a Bisheng filename"
#     fi

#     log "Downloading: ${BISHENG_FILENAME}"
#     curl -fL "${OBS_BASE}/cann/${BISHENG_FILENAME}" -o /tmp/bisheng.run

#     if [ -n "${BISHENG_SHA256}" ]; then
#       echo "${BISHENG_SHA256}  /tmp/bisheng.run" | sha256sum -c -
#     else
#       warn "No SHA256 checksum provided, skipping verification"
#     fi

#     chmod +x /tmp/bisheng.run
#     /tmp/bisheng.run --quiet --install --install-path=/usr/local/bisheng
#     export PATH="/usr/local/bisheng/tools/bishengir/bin:$PATH"
#     log "Bisheng installed"
#   fi
# else
#   warn "SKIP_BISHENG=1 — skipping Bisheng compiler install"
# fi

# ── Step 6: Cache build dependencies ────────────────────────────────────────
# NOTE: The GitHub Actions `actions/cache` step cannot be replicated in a
# plain shell script. The cache directories (~/.triton/llvm, ~/.triton/json,
# ~/.triton/nvidia, ~/.ccache) persist across local runs anyway, so as long as
# you run this script repeatedly on the same machine, the cache is effective.

# ── Step 7: Install TA_main2main_workflow ────────────────────────────────────
log "Installing TA_main2main_workflow"
if [ -d /tmp/ta-workflow ]; then
  rm -rf /tmp/ta-workflow
fi
git clone https://github.com/TecJesh/main2main_workflow.git /tmp/ta-workflow
cd /tmp/ta-workflow
${PYTHON} -m pip install -e .
log "TA workflow installed: $(ta-kickoff --help 2>&1 | head -1)"
cd "$TRITON_ASCEND_PATH"

# # ── Step 8: Install Claude Code CLI ──────────────────────────────────────────
# if [ "${SKIP_CLAUDE:-0}" != "1" ]; then
#   log "Installing Claude Code CLI"
#   if ! command -v npm >/dev/null 2>&1; then
#     NODE_VER="v20.18.0"
#     NODE_DIR="node-${NODE_VER}-linux-arm64"
#     ARCH=$(uname -m)
#     if [ "$ARCH" = "x86_64" ]; then
#       NODE_DIR="node-${NODE_VER}-linux-x64"
#     fi
#     log "Installing ${NODE_DIR} from Huawei mirror..."
#     curl -fL "https://repo.huaweicloud.com/nodejs/${NODE_VER}/${NODE_DIR}.tar.xz" -o /tmp/node.tar.xz
#     mkdir -p /usr/local/lib/nodejs
#     tar -xJf /tmp/node.tar.xz -C /usr/local/lib/nodejs
#     export PATH="/usr/local/lib/nodejs/${NODE_DIR}/bin:$PATH"
#     log "Installed node=$(node -v) npm=$(npm -v)"
#   else
#     log "Node already present: node=$(node -v 2>/dev/null) npm=$(npm -v 2>/dev/null)"
#   fi

#   npm install -g @anthropic-ai/claude-code
#   log "claude: $(command -v claude || echo '<not found>')"
# else
#   warn "SKIP_CLAUDE=1 — skipping Claude Code CLI install"
# fi

# ── Step 9: Clone llvm-project ──────────────────────────────────────────────
if [ "${SKIP_LLVM:-0}" != "1" ]; then
  log "Cloning llvm-project (this may take a while)"
  git config --global http.postBuffer 524288000
  git config --global http.maxRequestBuffer 524288000
  git config --global http.lowSpeedLimit 0
  git config --global http.lowSpeedTime 999999
  git config --global pack.windowMemory 256m
  git config --global pack.packSizeLimit 2g
  git config --global protocol.version 2

  # Retry helper
  retry() {
    for i in 1 2 3 4 5 6; do
      echo "[attempt $i/6]" "$@"
      if "$@"; then
        return 0
      fi
      if ! git status &>/dev/null; then
        echo "Repo corrupted, removing $LLVM_PROJECT_PATH for fresh clone"
        cd / && rm -rf "$LLVM_PROJECT_PATH"
        return 1
      fi
      echo "Retrying in 20s..."
      sleep 20
    done
    echo "ERROR: command failed after 6 attempts: $*" >&2
    return 1
  }

  if [ ! -d "$LLVM_PROJECT_PATH/.git" ]; then
    retry git clone --depth=1 https://github.com/llvm/llvm-project.git "$LLVM_PROJECT_PATH"
  fi
  log "llvm-project ready at $LLVM_PROJECT_PATH ($(git -C "$LLVM_PROJECT_PATH" rev-parse --short HEAD))"
else
  warn "SKIP_LLVM=1 — skipping llvm-project clone"
fi

# ── Step 10: Prepare LLVM install prefix ────────────────────────────────────
mkdir -p "$LLVM_INSTALL_PREFIX_SYNC"

# ── Step 11: Source CANN environment ────────────────────────────────────────
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
  source /usr/local/Ascend/cann/set_env.sh || warn "CANN set_env.sh returned non-zero"
else
  warn "/usr/local/Ascend/cann/set_env.sh not found — CANN env may be missing"
fi

export PATH="/usr/local/bisheng/tools/bishengir/bin:$PATH"

# Show NPU status
npu-smi info 2>/dev/null || warn "npu-smi not available"

# ── Step 12: Run the full sync flow ─────────────────────────────────────────
log "Starting ta-kickoff --mode=full"
log "  TRITON_ASCEND_PATH=$TRITON_ASCEND_PATH"
log "  TRITON_PATH=$TRITON_PATH"
log "  TARGET_COMMIT=${TARGET_COMMIT:-<latest>}"
log "  TA_MAIN2MAIN_WORKSPACE=$TA_MAIN2MAIN_WORKSPACE"

# Limit build parallelism (matching integration-tests-ascend.yml)
MAX_JOBS=$(( $(nproc) / 3 ))
[ "$MAX_JOBS" -lt 1 ] && MAX_JOBS=1
NUM_PROCS=$(( $(nproc) / 15 ))
[ "$NUM_PROCS" -lt 1 ] && NUM_PROCS=1
export MAX_JOBS NUM_PROCS
log "MAX_JOBS=$MAX_JOBS  NUM_PROCS=$NUM_PROCS"

# Build / test / AI environment variables (mirrors the workflow's env block)
export SKIP_BASELINE_LLVM="true"
export PYTHON="python3.11"
export TRITON_ASCEND_PATH
export TRITON_PATH
export TRITON_TARGET_COMMIT="${TARGET_COMMIT:-}"
export TA_MAIN2MAIN_WORKSPACE
export LLVM_PROJECT_PATH
export LLVM_INSTALL_PREFIX_SYNC
export TA_SINGLE_STEP_MODE="true"
export PUSH_TO_GITHUB="true"
export GITHUB_REPO="TecJesh/triton-ascend"
export PR_AUTHOR="TA"
export PR_TYPE="sync"
export AI_BACKEND="claude"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.deepseek.com/anthropic}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-deepseek-v4-pro[1m]}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${ANTHROPIC_DEFAULT_OPUS_MODEL:-deepseek-v4-pro[1m]}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${ANTHROPIC_DEFAULT_SONNET_MODEL:-deepseek-v4-pro[1m]}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${ANTHROPIC_DEFAULT_HAIKU_MODEL:-deepseek-v4-flash}"
export CLAUDE_CODE_SUBAGENT_MODEL="${CLAUDE_CODE_SUBAGENT_MODEL:-deepseek-v4-flash}"
export CLAUDE_CODE_EFFORT_LEVEL="${CLAUDE_CODE_EFFORT_LEVEL:-max}"
export AUTO_STASH="true"
export TRITON_DISABLE_LINE_INFO="1"
export CCACHE_COMPRESS="true"
export TRITON_BUILD_WITH_O1="true"

ta-kickoff --mode=full

# ── Done ────────────────────────────────────────────────────────────────────
log "Sync completed successfully."
log "Workspace artifacts are in: $TA_MAIN2MAIN_WORKSPACE"
