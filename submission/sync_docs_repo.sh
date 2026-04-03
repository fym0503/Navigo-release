#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  submission/sync_docs_repo.sh /path/to/Navigo-docs

Syncs the docs source from Navigo-release/docs into a standalone docs repo.
The target repo is expected to already be a git repository.
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but not installed." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_docs="$repo_root/docs"
target_repo="$(cd "$1" && pwd)"

if [[ ! -d "$source_docs" ]]; then
  echo "Source docs directory not found: $source_docs" >&2
  exit 1
fi

if [[ ! -e "$target_repo/.git" ]]; then
  echo "Target is not a git repository: $target_repo" >&2
  exit 1
fi

echo "Syncing docs from $source_docs to $target_repo"

rsync -a --delete \
  --exclude '.git' \
  --exclude '.git/' \
  --exclude '.venv*/' \
  --exclude '_build/' \
  --exclude '__pycache__/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude '.gitignore' \
  --exclude '.readthedocs.yaml' \
  --exclude 'README.md' \
  --exclude '*.executed.ipynb' \
  --exclude 'tutorials/outputs/' \
  --exclude 'tutorials/resources/interpolation/outputs/' \
  "$source_docs"/ "$target_repo"/

cat > "$target_repo/.readthedocs.yaml" <<'EOF'
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
  configuration: conf.py

python:
  install:
    - requirements: requirements.txt
EOF

cat > "$target_repo/.gitignore" <<'EOF'
_build/
.venv*/
.ipynb_checkpoints/
__pycache__/
*.executed.ipynb
tutorials/outputs/
tutorials/resources/interpolation/outputs/
EOF

cat > "$target_repo/README.md" <<'EOF'
# Navigo Documentation

This repository contains the standalone Sphinx/MyST documentation site for the Navigo project.

## Local build

```bash
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

## Read the Docs

Read the Docs is configured through `.readthedocs.yaml` at the repository root.
EOF

python3 <<'PY'
import os
import re
from pathlib import Path

target = Path(os.environ["TARGET_REPO"])


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


conf_path = target / "conf.py"
conf_text = conf_path.read_text(encoding="utf-8")
conf_text = re.sub(
    r'exclude_patterns\s*=\s*\[[^\]]*\]',
    """exclude_patterns = [
    "_build",
    ".git",
    ".venv*",
    "README.md",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "tutorials/README.md",
    "tutorials/resources/knockout/gene_compensation_clean_github/notebooks/01_Limb_Mesenchyme_Pathway_Heatmap_Replication.ipynb",
]""",
    conf_text,
    count=1,
    flags=re.S,
)
write(conf_path, conf_text)

write(
    target / "installation.md",
    """# Installation

## Package install

Install the Navigo package from the main code repository, not from this documentation repository.

```bash
git clone <your-github-url> Navigo-release
cd Navigo-release
pip install -r requirements.txt
pip install -e .
```

## Build the documentation locally

```bash
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

## Repository notes

- This repository contains the standalone documentation website.
- Tutorial notebooks live in `tutorials/`.
- Runtime data and checkpoints are expected from the main `Navigo-release` workspace.
""",
)

tutorial_index_path = target / "tutorials" / "index.md"
tutorial_index_text = tutorial_index_path.read_text(encoding="utf-8")
tutorial_index_text = tutorial_index_text.replace(
    "In this unified repository, tutorial notebooks live under `docs/tutorials/notebooks/`, while shared inputs are centralized at the repository root in `data/` and `checkpoints/`.",
    "In this unified repository, tutorial notebooks live under `tutorials/notebooks/`, while runtime data and checkpoints are expected from the main `Navigo-release` workspace.",
)
write(tutorial_index_path, tutorial_index_text)

edit_colab_url_path = target / "extensions" / "edit_colab_url.py"
edit_colab_url_text = edit_colab_url_path.read_text(encoding="utf-8")
edit_colab_url_text = edit_colab_url_text.replace(
    '"/docs/tutorials/notebooks"',
    '"/tutorials/notebooks"',
)
write(edit_colab_url_path, edit_colab_url_text)
PY

echo "Sync complete."
echo "Next steps:"
echo "  cd $target_repo"
echo "  git status"
