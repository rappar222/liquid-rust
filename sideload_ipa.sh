#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IPA_PATH="${1:-$SCRIPT_DIR/dist/liquid-rust-ios.ipa}"
ALTSERVER_BIN="${ALTSERVER_BIN:-$HOME/.local/bin/altserver}"
export ALTSERVER_ANISETTE_SERVER="${ALTSERVER_ANISETTE_SERVER:-https://ani.sidestore.io}"

if [[ ! -f "$IPA_PATH" ]]; then
    echo "ERROR: IPA file not found at: $IPA_PATH" >&2
    exit 1
fi

if [[ ! -x "$ALTSERVER_BIN" ]]; then
    if command -v altserver >/dev/null 2>&1; then
        ALTSERVER_BIN="$(command -v altserver)"
    elif command -v AltServer >/dev/null 2>&1; then
        ALTSERVER_BIN="$(command -v AltServer)"
    else
        echo "ERROR: AltServer-Linux binary not found at $ALTSERVER_BIN" >&2
        exit 1
    fi
fi

# Ensure usbmuxd is installed & active before talking to an attached iPhone/iPad
if ! pgrep -x usbmuxd >/dev/null 2>&1; then
    echo "NOTE: 'usbmuxd' is not running yet."
    echo "Run once on Arch Linux before plugging in your iPhone:"
    echo "  sudo pacman -S --needed usbmuxd libimobiledevice && sudo systemctl start usbmuxd"
    echo ""
fi

UDID="${IOS_UDID:-}"
if [[ -z "$UDID" ]] && command -v idevice_id >/dev/null 2>&1; then
    UDID="$(idevice_id -l 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$UDID" ]]; then
    echo "ERROR: No connected iOS device detected via USB."
    echo "1. Connect your iPhone/iPad via USB and tap 'Trust This Computer'."
    echo "2. Verify with: idevice_id -l"
    echo "3. Re-run: $0"
    exit 1
fi

echo "=== Liquid Rust iOS Sideloader (AltServer-Linux) ==="
echo "Detected Device UDID : $UDID"
echo "IPA Package          : $IPA_PATH"
echo "Anisette Server      : $ALTSERVER_ANISETTE_SERVER"
echo ""

APPLE_ID="${APPLE_ID:-}"
APPLE_PASSWORD="${APPLE_PASSWORD:-}"

if [[ -z "$APPLE_ID" ]]; then
    read -r -p "Apple ID (email): " APPLE_ID
fi
if [[ -z "$APPLE_PASSWORD" ]]; then
    read -r -s -p "Apple ID Password: " APPLE_PASSWORD
    echo ""
fi

exec "$ALTSERVER_BIN" \
    -u "$UDID" \
    -a "$APPLE_ID" \
    -p "$APPLE_PASSWORD" \
    "$IPA_PATH"
