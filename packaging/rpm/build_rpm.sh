#!/usr/bin/env bash
# Build Fedora/RHEL .rpm package for liquid-rust
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION=$(grep -m1 '^version' "$ROOT_DIR/Cargo.toml" | cut -d '"' -f 2)
ARCH="${1:-aarch64}"

echo "Building RPM package for liquid-rust v${VERSION} (${ARCH})..."

mkdir -p "$ROOT_DIR/dist"

export PATH="$HOME/.cargo/bin:/tmp/zig-bin:$PATH"

if ! command -v cargo-generate-rpm >/dev/null 2>&1; then
    echo "Installing cargo-generate-rpm..."
    cargo install cargo-generate-rpm
fi

case "$ARCH" in
    aarch64|arm64)
        TARGET="aarch64-unknown-linux-gnu"
        RPM_ARCH="aarch64"
        ;;
    x86_64|amd64)
        TARGET="x86_64-unknown-linux-gnu"
        RPM_ARCH="x86_64"
        ;;
    all)
        "$0" aarch64
        "$0" x86_64
        exit 0
        ;;
    *)
        echo "Unknown architecture: $ARCH (supported: aarch64, x86_64, all)"
        exit 1
        ;;
esac

BIN_SRC="$ROOT_DIR/target/${TARGET}/release/liquid-rust"
if [ ! -f "$BIN_SRC" ]; then
    echo "Binary not found at $BIN_SRC. Compiling with cargo-zigbuild..."
    cargo zigbuild --release --target "${TARGET}.2.17"
fi

RPM_FILE="$ROOT_DIR/dist/liquid-rust-${VERSION}-1.${RPM_ARCH}.rpm"

cargo generate-rpm \
    --target "$TARGET" \
    --arch "$RPM_ARCH" \
    --auto-req disabled \
    -o "$RPM_FILE"

echo "RPM package successfully created at: $RPM_FILE"
