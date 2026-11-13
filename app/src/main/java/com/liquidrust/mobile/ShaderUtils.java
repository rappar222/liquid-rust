package com.liquidrust.mobile;

import android.content.res.AssetManager;
import android.opengl.GLES30;
import android.opengl.GLES31;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * Compiles and links GLSL ES 3.00 / 3.10 vertex, fragment, and compute shader programs from APK assets.
 */
public final class ShaderUtils {
    private static final String TAG = "LiquidRustShader";

    private ShaderUtils() {}

    public static String loadAssetText(AssetManager assets, String path) {
        StringBuilder sb = new StringBuilder(4096);
        try (InputStream is = assets.open(path);
             BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line).append('\n');
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to read shader asset: " + path, e);
        }
        return sb.toString();
    }

    public static String injectDefines(String src, String definesBlock) {
        if (definesBlock == null || definesBlock.isEmpty()) {
            return src;
        }
        int firstNewline = src.indexOf('\n');
        if (firstNewline > 0 && src.startsWith("#version")) {
            return src.substring(0, firstNewline + 1) + definesBlock + "\n" + src.substring(firstNewline + 1);
        }
        return definesBlock + "\n" + src;
    }

    public static int compileShader(int type, String source, String debugLabel) {
        int shader = GLES30.glCreateShader(type);
        if (shader == 0) {
            throw new RuntimeException("glCreateShader returned 0 for " + debugLabel);
        }
        GLES30.glShaderSource(shader, source);
        GLES30.glCompileShader(shader);

        int[] compiled = new int[1];
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            String log = GLES30.glGetShaderInfoLog(shader);
            GLES30.glDeleteShader(shader);
            Log.e(TAG, "Compile failed [" + debugLabel + "]:\n" + log);
            throw new RuntimeException("Shader compile error (" + debugLabel + "): " + log);
        }
        return shader;
    }

    public static int buildRenderProgram(AssetManager assets, String vertAsset, String fragAsset, String fragDefines) {
        String vsSrc = loadAssetText(assets, vertAsset);
        String fsSrc = injectDefines(loadAssetText(assets, fragAsset), fragDefines);

        int vs = compileShader(GLES30.GL_VERTEX_SHADER, vsSrc, vertAsset);
        int fs = compileShader(GLES30.GL_FRAGMENT_SHADER, fsSrc, fragAsset + (fragDefines != null ? " " + fragDefines : ""));

        int prog = GLES30.glCreateProgram();
        GLES30.glAttachShader(prog, vs);
        GLES30.glAttachShader(prog, fs);
        GLES30.glLinkProgram(prog);

        GLES30.glDeleteShader(vs);
        GLES30.glDeleteShader(fs);

        int[] linked = new int[1];
        GLES30.glGetProgramiv(prog, GLES30.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) {
            String log = GLES30.glGetProgramInfoLog(prog);
            GLES30.glDeleteProgram(prog);
            Log.e(TAG, "Program link failed [" + vertAsset + "]:\n" + log);
            throw new RuntimeException("Program link error: " + log);
        }
        return prog;
    }

    public static int buildComputeProgram(AssetManager assets, String compAsset) {
        String csSrc = loadAssetText(assets, compAsset);
        int cs = compileShader(GLES31.GL_COMPUTE_SHADER, csSrc, compAsset);

        int prog = GLES31.GL_COMPUTE_SHADER != 0 ? GLES30.glCreateProgram() : 0;
        GLES30.glAttachShader(prog, cs);
        GLES30.glLinkProgram(prog);
        GLES30.glDeleteShader(cs);

        int[] linked = new int[1];
        GLES30.glGetProgramiv(prog, GLES30.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) {
            String log = GLES30.glGetProgramInfoLog(prog);
            GLES30.glDeleteProgram(prog);
            Log.e(TAG, "Compute link failed [" + compAsset + "]:\n" + log);
            throw new RuntimeException("Compute link error (" + compAsset + "): " + log);
        }
        return prog;
    }
}
