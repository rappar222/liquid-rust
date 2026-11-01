#!/usr/bin/env bash
set -e
echo "Configuring liquid-rust APT repository..."
echo "deb [arch=amd64 trusted=yes] https://rappar222.github.io/liquid-rust/ stable main" | tee /etc/apt/sources.list.d/liquid-rust.list
echo "Updating package lists..."
apt update
echo "Done! You can now run: sudo apt install liquid-rust"
