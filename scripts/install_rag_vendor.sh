#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REPO="https://github.com/sumner-harris/RAGAgent_AutonomousMaterialsSynthesis.git"
DEFAULT_REF="738d83d2ce7069cc35be69088522af39e68311e0"
DEFAULT_TARGET="vendor/RAGAgent_AutonomousMaterialsSynthesis"

REPO_URL="${RAG_VENDOR_REPO:-$DEFAULT_REPO}"
REF="${RAG_VENDOR_REF:-$DEFAULT_REF}"
TARGET="${RAG_VENDOR_TARGET:-$DEFAULT_TARGET}"
INSTALL_REQUIREMENTS=0

usage() {
  cat <<'EOF'
Usage: scripts/install_rag_vendor.sh [options]

Clone or update the optional RAGAgent_AutonomousMaterialsSynthesis vendor
checkout used by [rag].package_path.

Options:
  --repo URL                 Git repository to clone.
  --ref REF                  Commit, tag, or branch to check out.
  --target PATH              Destination path.
  --install-requirements     Run python -m pip install -r requirements.txt after checkout.
  -h, --help                 Show this help.

Environment overrides:
  RAG_VENDOR_REPO            Same as --repo.
  RAG_VENDOR_REF             Same as --ref.
  RAG_VENDOR_TARGET          Same as --target.

Defaults:
  repo:   https://github.com/sumner-harris/RAGAgent_AutonomousMaterialsSynthesis.git
  ref:    738d83d2ce7069cc35be69088522af39e68311e0
  target: vendor/RAGAgent_AutonomousMaterialsSynthesis
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo)
      REPO_URL="${2:?--repo requires a URL}"
      shift 2
      ;;
    --ref)
      REF="${2:?--ref requires a ref}"
      shift 2
      ;;
    --target)
      TARGET="${2:?--target requires a path}"
      shift 2
      ;;
    --install-requirements)
      INSTALL_REQUIREMENTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but was not found on PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$TARGET")"

if [ -e "$TARGET" ] && [ ! -d "$TARGET/.git" ]; then
  echo "Refusing to overwrite non-git path: $TARGET" >&2
  echo "Remove it, move it aside, or pass --target to use a different path." >&2
  exit 1
fi

if [ -d "$TARGET/.git" ]; then
  echo "Updating existing RAG vendor checkout at $TARGET"
  git -C "$TARGET" remote set-url origin "$REPO_URL"
  git -C "$TARGET" fetch --tags --prune origin
else
  echo "Cloning RAG vendor checkout into $TARGET"
  git clone "$REPO_URL" "$TARGET"
  git -C "$TARGET" fetch --tags --prune origin
fi

if git -C "$TARGET" rev-parse --verify --quiet "$REF^{commit}" >/dev/null; then
  git -C "$TARGET" checkout --detach "$REF"
elif git -C "$TARGET" rev-parse --verify --quiet "origin/$REF^{commit}" >/dev/null; then
  git -C "$TARGET" checkout --detach "origin/$REF"
else
  echo "Could not resolve ref '$REF' in $REPO_URL" >&2
  exit 1
fi

if [ -f "$TARGET/.gitmodules" ]; then
  git -C "$TARGET" submodule update --init --recursive
fi

if [ "$INSTALL_REQUIREMENTS" -eq 1 ]; then
  PYTHON_BIN="${PYTHON:-python}"
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "$PYTHON_BIN is required for --install-requirements but was not found on PATH" >&2
    exit 1
  fi
  if [ ! -f "$TARGET/requirements.txt" ]; then
    echo "No requirements.txt found at $TARGET/requirements.txt" >&2
    exit 1
  fi
  "$PYTHON_BIN" -m pip install -r "$TARGET/requirements.txt"
fi

resolved_ref="$(git -C "$TARGET" rev-parse HEAD)"
echo "RAG vendor ready: $TARGET @ $resolved_ref"
printf 'Set [rag].enabled = true and [rag].package_path = "%s" to use it.\n' "$TARGET"
