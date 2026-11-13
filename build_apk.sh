#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer Android Studio's bundled JBR if available for deterministic d8/apksigner execution
if [[ -z "${JAVA_HOME:-}" && -x "/opt/android-studio/jbr/bin/javac" ]]; then
    export JAVA_HOME="/opt/android-studio/jbr"
    export PATH="$JAVA_HOME/bin:$PATH"
fi

SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Android/Sdk}}"
if [[ ! -d "$SDK_ROOT" ]]; then
    echo "ERROR: Android SDK not found at '$SDK_ROOT'. Set ANDROID_SDK_ROOT." >&2
    exit 1
fi

# Pick latest available build-tools and platform android.jar
BUILD_TOOLS_DIR="$(find "$SDK_ROOT/build-tools" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
if [[ -f "$SDK_ROOT/platforms/android-34/android.jar" ]]; then
    PLATFORM_JAR="$SDK_ROOT/platforms/android-34/android.jar"
else
    PLATFORM_JAR="$(find "$SDK_ROOT/platforms" -name "android.jar" | sort -V | tail -n 1)"
fi

AAPT2="$BUILD_TOOLS_DIR/aapt2"
D8="$BUILD_TOOLS_DIR/d8"
ZIPALIGN="$BUILD_TOOLS_DIR/zipalign"
APKSIGNER="$BUILD_TOOLS_DIR/apksigner"

echo "=== LiquidRust Android Release APK Builder ==="
echo "SDK Root    : $SDK_ROOT"
echo "Build Tools : $BUILD_TOOLS_DIR"
echo "Platform    : $PLATFORM_JAR"
echo "Java        : $(javac -version 2>&1)"

# Optional offline Khronos GLSL ES shader validation if glslangValidator is installed
if command -v glslangValidator >/dev/null 2>&1; then
    echo "[1/7] Validating OpenGL ES 3.10 Compute & 3.00 Render Shaders..."
    for comp in app/src/main/assets/shaders/*.comp; do
        glslangValidator "$comp" >/dev/null
    done
    glslangValidator -l app/src/main/assets/shaders/line.vert app/src/main/assets/shaders/line.frag >/dev/null
    glslangValidator -l app/src/main/assets/shaders/particle.vert app/src/main/assets/shaders/particle.frag >/dev/null
    glslangValidator -DMODE_3D=1 -l app/src/main/assets/shaders/particle.vert app/src/main/assets/shaders/particle.frag >/dev/null
else
    echo "[1/7] Skipping offline glslangValidator check (not installed)"
fi

BUILD_DIR="$SCRIPT_DIR/build/cli"
DIST_DIR="$SCRIPT_DIR/dist"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/gen" "$BUILD_DIR/classes" "$BUILD_DIR/dex" "$DIST_DIR"

echo "[2/7] Compiling Android resources with aapt2..."
"$AAPT2" compile \
    --dir app/src/main/res \
    -o "$BUILD_DIR/res.zip"

echo "[3/7] Linking manifest, resources & GLSL shader assets..."
"$AAPT2" link \
    -o "$BUILD_DIR/unaligned.apk" \
    -I "$PLATFORM_JAR" \
    --manifest app/src/main/AndroidManifest.xml \
    --java "$BUILD_DIR/gen" \
    -A app/src/main/assets \
    --min-sdk-version 26 \
    --target-sdk-version 34 \
    --auto-add-overlay \
    "$BUILD_DIR/res.zip"

echo "[4/7] Compiling Java sources (--release 17)..."
mapfile -t JAVA_SOURCES < <(find "$BUILD_DIR/gen" app/src/main/java -name "*.java" | sort)
javac \
    --release 17 \
    -encoding UTF-8 \
    -Xlint:unchecked \
    -cp "$PLATFORM_JAR" \
    -d "$BUILD_DIR/classes" \
    "${JAVA_SOURCES[@]}"

echo "[5/7] Compiling optimized release DEX bytecode with d8..."
mapfile -t CLASS_FILES < <(find "$BUILD_DIR/classes" -name "*.class" | sort)
"$D8" \
    --release \
    --min-api 26 \
    --lib "$PLATFORM_JAR" \
    --output "$BUILD_DIR/dex" \
    "${CLASS_FILES[@]}"

# Add classes.dex to root of APK archive
(
    cd "$BUILD_DIR/dex"
    zip -q -j "$BUILD_DIR/unaligned.apk" classes.dex
)

echo "[6/7] Page-aligning APK with zipalign..."
"$ZIPALIGN" -f -p 4 "$BUILD_DIR/unaligned.apk" "$BUILD_DIR/aligned.apk"

KEYSTORE="$SCRIPT_DIR/build/release-signing.jks"
if [[ ! -f "$KEYSTORE" ]]; then
    keytool -genkeypair \
        -keystore "$KEYSTORE" \
        -alias liquidrust \
        -keyalg RSA \
        -keysize 2048 \
        -validity 10000 \
        -storepass liquidrust123 \
        -keypass liquidrust123 \
        -dname "CN=LiquidRust Studio, OU=Graphics, O=LiquidRust, L=Bucharest, ST=RO, C=RO" >/dev/null 2>&1
fi

OUT_APK="$DIST_DIR/liquid-rust-android.apk"
echo "[7/7] Signing APK (v1 + v2 + v3 schemes) & verifying..."
"$APKSIGNER" sign \
    --ks "$KEYSTORE" \
    --ks-key-alias liquidrust \
    --ks-pass pass:liquidrust123 \
    --key-pass pass:liquidrust123 \
    --v1-signing-enabled true \
    --v2-signing-enabled true \
    --v3-signing-enabled true \
    --out "$OUT_APK" \
    "$BUILD_DIR/aligned.apk"

"$APKSIGNER" verify --verbose "$OUT_APK" >/dev/null
rm -rf "$BUILD_DIR"

APK_SIZE="$(du -h "$OUT_APK" | cut -f1)"
echo "=============================================="
echo "SUCCESS! Built optimized signed APK:"
echo "  Path : $OUT_APK"
echo "  Size : $APK_SIZE"
echo "  Install via ADB: adb install -r \"$OUT_APK\""
echo "=============================================="
