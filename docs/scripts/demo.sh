#!/bin/sh
# Regenerate docs/assets/demo.png — a freeze render of the README get-started command.
set -eu
cd "$(dirname "$0")/../.."
freeze --execute "env -u VIRTUAL_ENV uv run athome --help" \
  --theme dracula --font.size 15 --padding 20 \
  --output docs/assets/demo.png
