#!/usr/bin/env bash
# Build portable standalone AppImage for liquid-rust
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION=$(grep -m1 '^version' "$ROOT_DIR/Cargo.toml" | cut -d '"' -f 2)
APPDIR="$(mktemp -d /tmp/liquid-rust-appdir.XXXXXX)"
OUTPUT_APPIMAGE="$ROOT_DIR/dist/liquid-rust-x86_64.AppImage"

echo "Building AppImage for liquid-rust v${VERSION}..."

# Prefer glibc 2.17 zigbuild binary for maximum compatibility across all Linux distros
BIN_SRC="$ROOT_DIR/target/x86_64-unknown-linux-gnu/release/liquid-rust"
if [ ! -f "$BIN_SRC" ]; then
    BIN_SRC="$ROOT_DIR/target/release/liquid-rust"
fi

# Ensure icons exist
if [ ! -f "$ROOT_DIR/assets/icons/liquid-rust.png" ]; then
    rsvg-convert -w 256 -h 256 "$ROOT_DIR/assets/icons/liquid-rust.svg" -o "$ROOT_DIR/assets/icons/liquid-rust.png"
fi

# Create AppDir directory structure
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$APPDIR/usr/share/icons/hicolor/scalable/apps"

# Copy binary & assets
install -m 755 "$BIN_SRC" "$APPDIR/usr/bin/liquid-rust"
install -m 644 "$ROOT_DIR/assets/desktop/liquid-rust.desktop" "$APPDIR/liquid-rust.desktop"
install -m 644 "$ROOT_DIR/assets/desktop/liquid-rust.desktop" "$APPDIR/usr/share/applications/liquid-rust.desktop"
install -m 644 "$ROOT_DIR/assets/desktop/liquid-rust-2d.desktop" "$APPDIR/usr/share/applications/liquid-rust-2d.desktop"

install -m 644 "$ROOT_DIR/assets/icons/liquid-rust.png" "$APPDIR/liquid-rust.png"
install -m 644 "$ROOT_DIR/assets/icons/liquid-rust.png" "$APPDIR/.DirIcon"
install -m 644 "$ROOT_DIR/assets/icons/liquid-rust.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/liquid-rust.png"
install -m 644 "$ROOT_DIR/assets/icons/liquid-rust.svg" "$APPDIR/usr/share/icons/hicolor/scalable/apps/liquid-rust.svg"

# AppRun script
cat <<'EOF' > "$APPDIR/AppRun"
#!/bin/sh
SELF=$(readlink -f "$0")
HERE=${SELF%/*}
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
exec "${HERE}/usr/bin/liquid-rust" "$@"
EOF
chmod 755 "$APPDIR"
chmod -R a+rX "$APPDIR"
chmod a+rx "$APPDIR/AppRun"

mkdir -p "$ROOT_DIR/dist"

# Download appimagetool if not available
APPIMAGETOOL="/tmp/appimagetool"
if [ ! -f "$APPIMAGETOOL" ]; then
    curl -L -o "$APPIMAGETOOL" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$APPIMAGETOOL"
fi

# Run appimagetool
export ARCH=x86_64
"$APPIMAGETOOL" --appimage-extract-and-run "$APPDIR" "$OUTPUT_APPIMAGE"

rm -rf "$APPDIR"
echo "Successfully generated AppImage at: $OUTPUT_APPIMAGE"
