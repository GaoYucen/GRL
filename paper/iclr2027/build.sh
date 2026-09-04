#!/usr/bin/env bash
set -euo pipefail
PAPER=/workspace/GRL/paper/iclr2027
TOOLS=$PAPER/tools
CACHE=$PAPER/.cache
SRC=$PAPER/src/iclr2027
BUILD=$PAPER/build
mkdir -p "$BUILD" "$CACHE"
cd "$SRC"
if [ ! -x "$TOOLS/tectonic" ]; then
  echo "Missing $TOOLS/tectonic; run compiler setup first." >&2
  exit 2
fi
XDG_CACHE_HOME="$CACHE" timeout 180 "$TOOLS/tectonic" -X compile paper.tex --outdir "$BUILD"
cp -f "$BUILD/paper.pdf" "$BUILD/grl_iclr2027_draft.pdf"
echo "Built $BUILD/grl_iclr2027_draft.pdf"
