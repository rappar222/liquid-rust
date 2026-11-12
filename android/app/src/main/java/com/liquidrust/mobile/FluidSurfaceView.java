package com.liquidrust.mobile;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.os.Build;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.Surface;
import android.view.SurfaceHolder;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;

/**
 * High-refresh-rate OpenGL ES 3.2 / 3.1 / 3.0 SurfaceView with multi-touch pinch zoom,
 * two-finger pan, orbital 3D camera gestures, and direct single-finger fluid force rays.
 */
public final class FluidSurfaceView extends GLSurfaceView {

    private static final int EGL_CONTEXT_CLIENT_VERSION = 0x3098;
    private static final int EGL_CONTEXT_MINOR_VERSION_KHR = 0x30FB;
    private static final int EGL_OPENGL_ES3_BIT_KHR = 0x0040;

    private final SimulationConfig cfg;
    private final Camera3D camera;
    private final FluidRenderer renderer;
    private final ScaleGestureDetector scaleDetector;

    private float lastTouchX = 0f;
    private float lastTouchY = 0f;
    private float lastMidX = 0f;
    private float lastMidY = 0f;
    private boolean wasMultiTouch = false;

    public FluidSurfaceView(Context context, SimulationConfig cfg, FluidRenderer.TelemetryListener listener) {
        super(context);
        this.cfg = cfg;
        this.camera = new Camera3D();

        setEGLConfigChooser((egl, display) -> {
            int[] attrList = {
                EGL10.EGL_RED_SIZE, 8,
                EGL10.EGL_GREEN_SIZE, 8,
                EGL10.EGL_BLUE_SIZE, 8,
                EGL10.EGL_ALPHA_SIZE, 8,
                EGL10.EGL_DEPTH_SIZE, 16,
                EGL10.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
                EGL10.EGL_NONE
            };
            EGLConfig[] configs = new EGLConfig[1];
            int[] numConfig = new int[1];
            if (egl.eglChooseConfig(display, attrList, configs, 1, numConfig) && numConfig[0] > 0) {
                return configs[0];
            }
            // Fallback ES2/ES3 generic config
            int[] fallback = {
                EGL10.EGL_RED_SIZE, 8,
                EGL10.EGL_GREEN_SIZE, 8,
                EGL10.EGL_BLUE_SIZE, 8,
                EGL10.EGL_DEPTH_SIZE, 16,
                EGL10.EGL_NONE
            };
            egl.eglChooseConfig(display, fallback, configs, 1, numConfig);
            return configs[0];
        });

        setEGLContextFactory(new EGLContextFactory() {
            @Override
            public EGLContext createContext(EGL10 egl, EGLDisplay display, EGLConfig config) {
                // Negotiate highest supported OpenGL ES context: 3.2 -> 3.1 -> 3.0
                int[][] candidates = {
                    { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_CONTEXT_MINOR_VERSION_KHR, 2, EGL10.EGL_NONE },
                    { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_CONTEXT_MINOR_VERSION_KHR, 1, EGL10.EGL_NONE },
                    { EGL_CONTEXT_CLIENT_VERSION, 3, EGL10.EGL_NONE }
                };
                for (int[] attrs : candidates) {
                    EGLContext ctx = egl.eglCreateContext(display, config, EGL10.EGL_NO_CONTEXT, attrs);
                    if (ctx != null && ctx != EGL10.EGL_NO_CONTEXT) {
                        return ctx;
                    }
                }
                return EGL10.EGL_NO_CONTEXT;
            }

            @Override
            public void destroyContext(EGL10 egl, EGLDisplay display, EGLContext context) {
                egl.eglDestroyContext(display, context);
            }
        });

        setPreserveEGLContextOnPause(true);

        this.renderer = new FluidRenderer(context.getAssets(), cfg, camera, listener);
        setRenderer(this.renderer);
        setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        this.scaleDetector = new ScaleGestureDetector(context, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override
            public boolean onScale(ScaleGestureDetector detector) {
                camera.zoomByFactor(detector.getScaleFactor());
                return true;
            }
        });
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        super.surfaceCreated(holder);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            try {
                Surface s = holder.getSurface();
                if (s != null && s.isValid()) {
                    s.setFrameRate(120.0f, Surface.FRAME_RATE_COMPATIBILITY_DEFAULT);
                }
            } catch (Throwable ignored) {}
        }
    }

    public void shutdown() {
        renderer.shutdown();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        scaleDetector.onTouchEvent(event);

        int action = event.getActionMasked();
        int pointerCount = event.getPointerCount();

        if (pointerCount >= 2) {
            cfg.activeTouchForceMode = 0;
            float mx = (event.getX(0) + event.getX(1)) * 0.5f;
            float my = (event.getY(0) + event.getY(1)) * 0.5f;

            if (!wasMultiTouch || action == MotionEvent.ACTION_POINTER_DOWN) {
                lastMidX = mx;
                lastMidY = my;
                wasMultiTouch = true;
            } else if (action == MotionEvent.ACTION_MOVE) {
                camera.pan(mx - lastMidX, my - lastMidY);
                lastMidX = mx;
                lastMidY = my;
            }
            return true;
        }

        if (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL) {
            cfg.activeTouchForceMode = 0;
            wasMultiTouch = false;
            return true;
        }

        if (wasMultiTouch) {
            // Smoothly re-anchor after lifting second finger
            lastTouchX = event.getX(0);
            lastTouchY = event.getY(0);
            wasMultiTouch = false;
            return true;
        }

        float x = event.getX(0);
        float y = event.getY(0);

        if (action == MotionEvent.ACTION_DOWN) {
            lastTouchX = x;
            lastTouchY = y;
            if (cfg.touchTool == SimulationConfig.TOOL_ATTRACT) {
                cfg.activeTouchForceMode = 1;
                camera.computeTouchRay(x, y, cfg.is2D, cfg);
            } else if (cfg.touchTool == SimulationConfig.TOOL_REPEL) {
                cfg.activeTouchForceMode = -1;
                camera.computeTouchRay(x, y, cfg.is2D, cfg);
            } else {
                cfg.activeTouchForceMode = 0;
            }
        } else if (action == MotionEvent.ACTION_MOVE) {
            float dx = x - lastTouchX;
            float dy = y - lastTouchY;
            lastTouchX = x;
            lastTouchY = y;

            if (cfg.touchTool == SimulationConfig.TOOL_ATTRACT) {
                cfg.activeTouchForceMode = 1;
                camera.computeTouchRay(x, y, cfg.is2D, cfg);
            } else if (cfg.touchTool == SimulationConfig.TOOL_REPEL) {
                cfg.activeTouchForceMode = -1;
                camera.computeTouchRay(x, y, cfg.is2D, cfg);
            } else {
                cfg.activeTouchForceMode = 0;
                if (cfg.is2D) {
                    camera.pan(dx, dy);
                } else {
                    camera.orbit(-dx * 0.0065f, dy * 0.0065f);
                }
            }
        }

        return true;
    }
}
