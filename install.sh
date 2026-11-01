#!/usr/bin/env bash
# liquid-rust installer script
# High-performance 3D SPH Fluid Simulator (GPU accelerated via Vulkan)

set -e

BOLD="\033[1m"
GREEN="\033[32m"
CYAN="\033[36m"
YELLOW="\033[33m"
RESET="\033[0m"

echo -e "${CYAN}${BOLD}"
echo "============================================================"
echo "          🌊 Liquid-Rust 3D SPH Fluid Simulator             "
echo "        GPU-Accelerated (Vulkan) Real-time Engine           "
echo "============================================================"
echo -e "${RESET}"

# Determine install directory
INSTALL_DIR="$HOME/.local/bin"
if [ "$EUID" -eq 0 ]; then
    INSTALL_DIR="/usr/local/bin"
fi
mkdir -p "$INSTALL_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
BINARY_SOURCE=""

# 1. Check if binary exists locally next to install.sh
if [ -f "$SCRIPT_DIR/liquid-rust" ]; then
    BINARY_SOURCE="$SCRIPT_DIR/liquid-rust"
elif [ -f "./target/release/liquid-rust" ]; then
    BINARY_SOURCE="./target/release/liquid-rust"
fi

# 2. If not local, download from GitHub Releases
if [ -z "$BINARY_SOURCE" ]; then
    echo -e "${YELLOW}Downloading latest liquid-rust release from GitHub...${RESET}"
    TMP_DIR=$(mktemp -d)
    TAR_URL="https://github.com/rappar222/liquid-rust/releases/latest/download/liquid-rust-linux-x86_64.tar.gz"
    
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$TAR_URL" -o "$TMP_DIR/liquid-rust.tar.gz"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$TMP_DIR/liquid-rust.tar.gz" "$TAR_URL"
    else
        echo "Error: neither curl nor wget is installed."
        exit 1
    fi

    tar -xzf "$TMP_DIR/liquid-rust.tar.gz" -C "$TMP_DIR"
    BINARY_SOURCE="$TMP_DIR/liquid-rust"
fi

# 3. Install binary
echo -e "${CYAN}Installing liquid-rust to: ${BOLD}$INSTALL_DIR/liquid-rust${RESET}"
cp -f "$BINARY_SOURCE" "$INSTALL_DIR/liquid-rust.tmp"
chmod +x "$INSTALL_DIR/liquid-rust.tmp"
mv -f "$INSTALL_DIR/liquid-rust.tmp" "$INSTALL_DIR/liquid-rust"

# 4. Also install to ~/bin if it exists in PATH
if [ -d "$HOME/bin" ]; then
    cp -f "$INSTALL_DIR/liquid-rust" "$HOME/bin/liquid-rust" 2>/dev/null || true
fi

# 5. Check PATH configuration
PATH_FOUND=0
case ":$PATH:" in
    *":$INSTALL_DIR:"*) PATH_FOUND=1 ;;
esac

if [ "$PATH_FOUND" -eq 0 ]; then
    echo -e "${YELLOW}Adding $INSTALL_DIR to your shell PATH...${RESET}"
    
    # Fish shell
    if [ -f "$HOME/.config/fish/config.fish" ]; then
        if ! grep -q "$INSTALL_DIR" "$HOME/.config/fish/config.fish" 2>/dev/null; then
            echo "fish_add_path $INSTALL_DIR" >> "$HOME/.config/fish/config.fish"
        fi
    fi
    
    # Bash
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "$INSTALL_DIR" "$HOME/.bashrc" 2>/dev/null; then
            echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$HOME/.bashrc"
        fi
    fi
    
    # Zsh
    if [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "$INSTALL_DIR" "$HOME/.zshrc" 2>/dev/null; then
            echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$HOME/.zshrc"
        fi
    fi
fi

# 6. Create Desktop launcher entry (for GUI application menu)
DESKTOP_DIR="$HOME/.local/share/applications"
mkdir -p "$DESKTOP_DIR"
cat <<EOF > "$DESKTOP_DIR/liquid-rust.desktop"
[Desktop Entry]
Name=Liquid-Rust
Comment=High-Performance 3D SPH Fluid Simulator (GPU Accelerated)
Exec=$INSTALL_DIR/liquid-rust
Terminal=false
Type=Application
Categories=Graphics;Science;Simulation;
EOF
chmod +x "$DESKTOP_DIR/liquid-rust.desktop"

echo -e "${GREEN}${BOLD}✔ liquid-rust installed successfully!${RESET}\n"
echo -e "${BOLD}How to run:${RESET}"
echo -e "  ${CYAN}• GUI Mode (Interactive 3D):${RESET}"
echo "      liquid-rust"
echo ""
echo -e "  ${CYAN}• CLI Headless Mode (MP4 Video Export):${RESET}"
echo "      liquid-rust --preset double-dam -q ultra --time 5.0 --export-mp4 splash.mp4"
echo ""
echo -e "  ${CYAN}• View Help & CLI Options:${RESET}"
echo "      liquid-rust --help"
echo ""
echo -e "  ${YELLOW}(Note: If command is not found in your current shell tab, run: export PATH=\"\$PATH:$INSTALL_DIR\" or open a new terminal)${RESET}"
