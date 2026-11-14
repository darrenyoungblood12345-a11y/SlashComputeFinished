#!/bin/bash
# Build a self-contained /compute.app and dist/compute-<version>-arm64.dmg.
#
# Unlike install_app.sh (which launches this checkout's venv), the app carries its own Python
# (uv's portable CPython), the locked dependencies, and, if found, llama.cpp for the LLMs tab.
#
#   ./scripts/macos/build_dmg.sh
#   LLAMA_BIN=/path/to/llama.cpp/build/bin ./scripts/macos/build_dmg.sh   # llama.cpp to bundle
#
# A build that must run beside another /compute on the same Mac (a newer generation of the app
# uses the same name, data folder, ports and Bonjour service) gets its own identity:
#
#   APP_NAME="compute classic" BUNDLE_ID=com.slashcompute.classic DATA_DIR=.slashcompute-classic \
#   SHELL_PORT=8776 COORDINATOR_PORT=8775 DATA_PORT=9710 MDNS_SERVICE=_slashcompute-classic._tcp \
#   DMG_FILE=dist/compute-classic-mac.dmg ./scripts/macos/build_dmg.sh
#
# Every Mac in a pool has to run the same build: these values are how they find each other.
# The app is ad-hoc signed, not notarized: on first open use right-click → Open.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="$(sed -nE 's/^version = "([^"]+)"/\1/p' "$REPO/pyproject.toml" | head -1)"
PY_VERSION="${PY_VERSION:-3.13}"
LLAMA_BIN="${LLAMA_BIN:-$REPO/vendor/llama.cpp/build/bin}"
DIST="$REPO/dist"
APP_NAME="${APP_NAME:-compute}"
DISPLAY_NAME="${DISPLAY_NAME:-/$APP_NAME}"
BUNDLE_ID="${BUNDLE_ID:-com.slashcompute.app}"
DATA_DIR="${DATA_DIR:-}"                  # folder under $HOME; empty = the app's default (~/.slashcompute)
SHELL_PORT="${SHELL_PORT:-}"              # local UI port; empty = 8766
COORDINATOR_PORT="${COORDINATOR_PORT:-}"  # empty = 8765
DATA_PORT="${DATA_PORT:-}"                # empty = 9700
MDNS_SERVICE="${MDNS_SERVICE:-_slashcompute._tcp}"
# Stage outside the checkout: iCloud-synced folders (e.g. ~/Desktop) re-tag files with
# Finder/provenance attributes that codesign rejects.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/compute-dmg.XXXXXX")"
STAGE="$WORK/dmg"
APP="$STAGE/$APP_NAME.app"
DMG="${DMG_FILE:-$DIST/${APP_NAME// /-}-$VERSION-arm64.dmg}"

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || { echo "Build on an Apple Silicon Mac." >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

CONTENTS="$APP/Contents"
RES="$CONTENTS/Resources"
rm -f "$DMG"
mkdir -p "$(dirname "$DMG")" "$CONTENTS/MacOS" "$RES"

# ------------------------------------------------------------ Python
echo "==> Python $PY_VERSION"
uv python install "$PY_VERSION" >/dev/null
# outside any project and venv, so uv returns its portable CPython rather than a .venv
PY_SRC="$(cd / && env -u VIRTUAL_ENV uv python find --managed-python --no-project "$PY_VERSION")"
case "$PY_SRC" in */.venv/*|*/venv/*) echo "refusing to bundle a virtualenv: $PY_SRC" >&2; exit 1 ;; esac
PY_HOME="$(cd "$(dirname "$PY_SRC")/.." && pwd -P)"
cp -R "$PY_HOME" "$RES/python"
PY="$RES/python/bin/python3"
rm -f "$RES"/python/lib/python3.*/EXTERNALLY-MANAGED

echo "==> dependencies (uv.lock) and slashcompute"
REQS="$(mktemp)"
trap 'rm -rf "$REQS" "$WORK"' EXIT
(cd "$REPO" && uv export --frozen --no-dev --no-emit-project --no-hashes --format requirements-txt -o "$REQS" >/dev/null)
uv pip install --python "$PY" --no-cache -q -r "$REQS"
uv pip install --python "$PY" --no-cache -q --no-deps "$REPO"

# installers may leave symlinks to the build machine's Python; keep the bundle self-contained
REAL_PY="$(find "$RES/python/bin" -maxdepth 1 -type f -name 'python3.*' ! -name '*-config' | head -1)"
for link in "$RES"/python/bin/*; do
  if [[ -L "$link" && "$(readlink "$link")" == /* ]]; then
    ln -sf "$(basename "$REAL_PY")" "$link"
  fi
done

SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# the worker sandbox profile ships inside the package; a missing one would stop training agents.
# objc/Foundation (PyObjC): the LLM node moves models removed from the pool to the Trash with it.
"$PY" -c "import slashcompute.launcher.main, slashcompute.inference.node.agent, mlx.core, webview, objc, Foundation
from slashcompute.agent.sandbox import profile_path
assert profile_path().is_file(), profile_path()" \
  || { echo "bundled Python cannot import the app" >&2; exit 1; }

# ------------------------------------------------------------ llama.cpp (LLMs tab)
if [[ -x "$LLAMA_BIN/llama-server" ]]; then
  echo "==> llama.cpp from $LLAMA_BIN"
  mkdir -p "$RES/llama"
  cp -a "$LLAMA_BIN"/llama-server "$LLAMA_BIN"/*.dylib "$RES/llama/"
  for rpc in ggml-rpc-server rpc-server; do
    if [[ -x "$LLAMA_BIN/$rpc" ]]; then cp -a "$LLAMA_BIN/$rpc" "$RES/llama/"; fi
  done
  # Binaries and dylibs find each other via @rpath: point it at their own folder.
  for f in "$RES"/llama/*; do
    if [[ -L "$f" ]]; then continue; fi
    otool -l "$f" | awk '/LC_RPATH/{getline; getline; print $2}' | while read -r old; do
      install_name_tool -delete_rpath "$old" "$f" 2>/dev/null || true
    done
    install_name_tool -add_rpath @loader_path "$f" 2>/dev/null || true
  done
  "$RES/llama/llama-server" --version >/dev/null 2>&1 || true
else
  echo "==> no llama.cpp at $LLAMA_BIN: the LLMs tab will need it on PATH (scripts/build_llama.sh)"
fi

# ------------------------------------------------------------ bundle
echo "==> app bundle"
"$PY" -m compileall -q "$SITE" >/dev/null 2>&1 || true

cat > "$CONTENTS/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleDisplayName</key><string>$DISPLAY_NAME</string>
  <key>CFBundleExecutable</key><string>compute</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSLocalNetworkUsageDescription</key><string>$DISPLAY_NAME finds and joins pools of Macs on your network.</string>
  <key>NSBonjourServices</key><array><string>$MDNS_SERVICE</string></array>
</dict>
</plist>
EOF

# What sets this build apart from another /compute on the same Mac. The app and everything it
# starts (coordinator, agent, worker, LLM node) read these from the environment.
IDENTITY=""
if [[ -n "$DATA_DIR" ]]; then IDENTITY+="export SLASHCOMPUTE_HOME=\"\$HOME/$DATA_DIR\""$'\n'; fi
if [[ -n "$SHELL_PORT" ]]; then IDENTITY+="export SLASHCOMPUTE_SHELL_PORT=$SHELL_PORT"$'\n'; fi
if [[ -n "$COORDINATOR_PORT" ]]; then IDENTITY+="export SLASHCOMPUTE_COORDINATOR_PORT=$COORDINATOR_PORT"$'\n'; fi
if [[ -n "$DATA_PORT" ]]; then IDENTITY+="export SLASHCOMPUTE_DATA_PORT=$DATA_PORT"$'\n'; fi
if [[ "$MDNS_SERVICE" != "_slashcompute._tcp" ]]; then IDENTITY+="export SLASHCOMPUTE_MDNS_SERVICE=$MDNS_SERVICE.local."$'\n'; fi

cat > "$CONTENTS/MacOS/compute" <<LAUNCHER
#!/bin/bash
RES="\$(cd "\$(dirname "\$0")/../Resources" && pwd)"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
if [[ -d "\$RES/llama" ]]; then export SLASHCOMPUTE_LLAMA_DIR="\$RES/llama"; fi
${IDENTITY}cd "\$HOME"
exec "\$RES/python/bin/python3" -m slashcompute.launcher.main
LAUNCHER
chmod +x "$CONTENTS/MacOS/compute"

ICONSET="$RES/AppIcon.iconset"
mkdir -p "$ICONSET"
for pair in 16:16 32:16 32:32 64:32 128:128 256:128 256:256 512:256 512:512 1024:512; do
  px="${pair%%:*}"; base="${pair##*:}"
  if [[ "$px" == "$base" ]]; then name="icon_${base}x${base}.png"; else name="icon_${base}x${base}@2x.png"; fi
  sips -z "$px" "$px" "$SCRIPT_DIR/icon.png" --out "$ICONSET/$name" >/dev/null
done
iconutil -c icns -o "$RES/AppIcon.icns" "$ICONSET"
rm -rf "$ICONSET"

echo "==> ad-hoc signing"
xattr -cr "$APP"   # codesign refuses Finder info / provenance attributes on copied files
find "$RES" -type f \( -name "*.so" -o -name "*.dylib" \) -exec codesign --force -s - {} \; 2>/dev/null
for f in "$RES"/python/bin/python3.* "$RES"/llama/llama-server "$RES"/llama/ggml-rpc-server "$RES"/llama/rpc-server; do
  if [[ -f "$f" && ! -L "$f" ]]; then codesign --force -s - "$f"; fi
done
codesign --force -s - "$APP"
codesign --verify --deep --strict "$APP"

# ------------------------------------------------------------ dmg
echo "==> $DMG"
ln -s /Applications "$STAGE/Applications"
hdiutil create -quiet -volname "$APP_NAME" -srcfolder "$STAGE" -ov -format UDZO "$WORK/compute.dmg"
mv "$WORK/compute.dmg" "$DMG"
echo
echo "Built $DMG ($(du -h "$DMG" | cut -f1))"
echo "Unsigned for Gatekeeper: on first open, right-click $APP_NAME → Open."
