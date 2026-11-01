#!/usr/bin/env bash
# Build Debian/Ubuntu .deb package for liquid-rust
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION=$(grep -m1 '^version' "$ROOT_DIR/Cargo.toml" | cut -d '"' -f 2)
ARCH="amd64"
DEB_NAME="liquid-rust_${VERSION}_${ARCH}.deb"
BUILD_DIR="$(mktemp -d /tmp/liquid-rust-deb.XXXXXX)"

echo "Building Debian package for liquid-rust v${VERSION} (${ARCH})..."

# Prefer glibc 2.17 zigbuild binary for maximum compatibility across all Linux distros
BIN_SRC="$ROOT_DIR/target/x86_64-unknown-linux-gnu/release/liquid-rust"
if [ ! -f "$BIN_SRC" ]; then
    BIN_SRC="$ROOT_DIR/target/release/liquid-rust"
fi

# Create directory structure
mkdir -p "$BUILD_DIR/DEBIAN"
mkdir -p "$BUILD_DIR/usr/bin"
mkdir -p "$BUILD_DIR/usr/share/applications"
mkdir -p "$BUILD_DIR/usr/share/icons/hicolor/scalable/apps"
mkdir -p "$BUILD_DIR/usr/share/doc/liquid-rust"

# Copy binary & assets
install -m 755 "$BIN_SRC" "$BUILD_DIR/usr/bin/liquid-rust"
install -m 644 "$ROOT_DIR/assets/desktop/liquid-rust.desktop" "$BUILD_DIR/usr/share/applications/"
install -m 644 "$ROOT_DIR/assets/desktop/liquid-rust-2d.desktop" "$BUILD_DIR/usr/share/applications/"
install -m 644 "$ROOT_DIR/assets/icons/liquid-rust.svg" "$BUILD_DIR/usr/share/icons/hicolor/scalable/apps/"
install -m 644 "$ROOT_DIR/assets/icons/liquid-rust-2d.svg" "$BUILD_DIR/usr/share/icons/hicolor/scalable/apps/"
install -m 644 "$ROOT_DIR/LICENSE" "$BUILD_DIR/usr/share/doc/liquid-rust/copyright"

# Generate control file
cat <<EOF > "$BUILD_DIR/DEBIAN/control"
Package: liquid-rust
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Maintainer: Radu <rappar2018@gmail.com>
Depends: libvulkan1, libx11-6, libx11-xcb1, libxcursor1, libxi6, libxrandr2, libxkbcommon0, libxkbcommon-x11-0
Homepage: https://github.com/rappar222/liquid-rust
Description: High-performance 2D/3D SPH fluid simulator in Rust
 Real-time GPU-accelerated fluid mechanics sandbox with interactive colliders,
 faucets, drains, and PLY/MP4 export.
EOF

mkdir -p "$ROOT_DIR/dist"
OUTPUT_DEB="$ROOT_DIR/dist/$DEB_NAME"

# Package into .deb using standard ar/tar (compatible across both Arch & Debian/Ubuntu)
tar -czf "$BUILD_DIR/control.tar.gz" -C "$BUILD_DIR/DEBIAN" .
tar -czf "$BUILD_DIR/data.tar.gz" -C "$BUILD_DIR" usr
echo "2.0" > "$BUILD_DIR/debian-binary"

ar rc "$OUTPUT_DEB" "$BUILD_DIR/debian-binary" "$BUILD_DIR/control.tar.gz" "$BUILD_DIR/data.tar.gz"
rm -rf "$BUILD_DIR"

echo "Package successfully generated at: $OUTPUT_DEB"
