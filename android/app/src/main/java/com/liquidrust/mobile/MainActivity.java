package com.liquidrust.mobile;

import android.animation.ValueAnimator;
import android.app.Activity;
import android.content.Context;
import android.content.res.ColorStateList;
import android.content.res.Configuration;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.DisplayMetrics;
import android.util.TypedValue;
import android.view.Display;
import android.view.Gravity;
import android.view.HapticFeedbackConstants;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.VelocityTracker;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.DecelerateInterpolator;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.window.OnBackInvokedCallback;
import android.window.OnBackInvokedDispatcher;

import java.util.Locale;

/**
 * Native Android Material Design 3 (Material You Dark) Activity for LiquidRust.
 * Features:
 * - Minimal top status pill + quick tonal circle buttons (zero text wrapping, zero emojis).
 * - Thumb-zone Material 3 segmented floating toolbar + squircle Play/Pause FAB.
 * - Interactive touch-draggable Material 3 Bottom Sheet with FilterChips, Liquid Swatches,
 *   animated M3 Pill Switches, and M3 Sliders.
 */
public final class MainActivity extends Activity implements SensorEventListener, FluidRenderer.TelemetryListener {

    // Material Design 3 Dark Color Tokens (Slate / Ocean Blue)
    private static final int M3_BG                  = 0xFF0E141B;
    private static final int M3_HUD_SURFACE         = 0xDC1A222D;
    private static final int M3_DOCK_SURFACE        = 0xEB1A222D;
    private static final int M3_SHEET_SURFACE       = 0xF7151C25;
    private static final int M3_CARD_SURFACE        = 0xFF1E2733;
    private static final int M3_TONAL_BTN           = 0xEB25303E;
    private static final int M3_PRIMARY             = 0xFF8ECDFF;
    private static final int M3_ON_PRIMARY          = 0xFF00344F;
    private static final int M3_PRIMARY_CONTAINER   = 0xFF004B70;
    private static final int M3_ON_PRIMARY_CONT     = 0xFFCBE6FF;
    private static final int M3_ON_SURFACE          = 0xFFE2E8F0;
    private static final int M3_ON_SURFACE_VAR      = 0xFF94A3B8;
    private static final int M3_OUTLINE             = 0xFF384657;
    private static final int M3_DIVIDER             = 0xFF263241;
    private static final int M3_RIPPLE              = 0x36CBE6FF;
    private static final int M3_DOT_GPU             = 0xFF4ADE80;
    private static final int M3_DOT_CPU             = 0xFFFBBF24;

    private static final Typeface TF_MEDIUM = Typeface.create("sans-serif-medium", Typeface.NORMAL);
    private static final Typeface TF_REGULAR = Typeface.create("sans-serif", Typeface.NORMAL);

    private SimulationConfig cfg;
    private FluidSurfaceView glView;

    private SensorManager sensorManager;
    private Sensor gravitySensor;
    private float smoothGx = 0.0f;
    private float smoothGy = -1.0f;
    private float smoothGz = 0.0f;

    // Top HUD
    private View statusDotView;
    private TextView tvStatusText;
    private boolean showDetailedStats = false;
    private String backendLabel = "GPU";
    private int lastFps = 60;
    private int lastParticles = 0;
    private float lastStepMs = 0.0f;

    private LinearLayout btnTopDimension;
    private MaterialIconDrawable iconTopDimension;
    private TextView tvTopDimension;

    private FrameLayout btnTopTilt;
    private MaterialIconDrawable iconTopTilt;

    // Bottom Thumb Dock
    private LinearLayout bottomDock;
    private SegmentView segPull;
    private SegmentView segPush;
    private SegmentView segOrbit;
    private FrameLayout fabPlayPause;
    private MaterialIconDrawable iconFabPlay;

    // Draggable Material 3 Bottom Sheet
    private LinearLayout bottomSheet;
    private ScrollView sheetScrollView;
    private boolean isSheetOpen = false;
    private int safeBottomInset = 0;
    private int safeTopInset = 0;
    private OnBackInvokedCallback backCallback;

    // Sheet Tabs
    private TabItemView tabScene;
    private TabItemView tabPhysics;
    private TabItemView tabAppearance;
    private LinearLayout paneScene;
    private LinearLayout panePhysics;
    private LinearLayout paneAppearance;

    // Synced Controls inside Bottom Sheet
    private FilterChipView[] presetChips;
    private FilterChipView[] qualityChips;
    private FilterChipView chipDim2D, chipDim3D;
    private TextView tvKernelInfo;

    private FilterChipView chipSpeedSlow, chipSpeedSmooth, chipSpeedReal;
    private SeekBar sbSpeed;
    private TextView tvSpeedBadge;

    private M3Switch swObstacle;
    private LinearLayout obstacleSlidersWrap;

    private M3Switch swTiltGravity;
    private SeekBar sbGravity;
    private TextView tvGravityBadge;

    private SwatchChipView[] materialSwatchChips;
    private FilterChipView[] colorModeChips;
    private LinearLayout colorRampWrap;
    private FilterChipView[] colorRampChips;
    private FilterChipView[] texModeChips;

    private SeekBar sbIor, sbRough, sbSpec, sbRefr, sbMetal, sbFoam;
    private TextView tvIorVal, tvRoughVal, tvSpecVal, tvRefrVal, tvMetalVal, tvFoamVal;
    private boolean suppressUiSync = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        configureImmersiveHighRefreshWindow();

        cfg = new SimulationConfig();
        cfg.isPortrait = (getResources().getConfiguration().orientation != Configuration.ORIENTATION_LANDSCAPE);
        glView = new FluidSurfaceView(this, cfg, this);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            gravitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
            if (gravitySensor == null) {
                gravitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            }
        }

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(M3_BG);
        root.addView(glView, new FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ));

        View topOverlay = buildTopOverlay();
        bottomDock = buildBottomThumbDock();
        bottomSheet = buildMaterialBottomSheet();

        FrameLayout.LayoutParams topLp = new FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.TOP
        );
        topLp.setMargins(dp(16), dp(12), dp(16), 0);
        root.addView(topOverlay, topLp);

        FrameLayout.LayoutParams dockLp = new FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM
        );
        dockLp.setMargins(dp(16), 0, dp(16), dp(16));
        root.addView(bottomDock, dockLp);

        FrameLayout.LayoutParams sheetLp = createBottomSheetLayoutParams(getResources().getConfiguration());
        root.addView(bottomSheet, sheetLp);

        root.setOnApplyWindowInsetsListener((v, insets) -> {
            safeTopInset = insets.getSystemWindowInsetTop();
            safeBottomInset = insets.getSystemWindowInsetBottom();

            topLp.topMargin = Math.max(dp(10), safeTopInset + dp(8));
            dockLp.bottomMargin = Math.max(dp(14), safeBottomInset + dp(12));
            topOverlay.setLayoutParams(topLp);
            bottomDock.setLayoutParams(dockLp);

            if (sheetScrollView != null) {
                sheetScrollView.setPadding(dp(16), dp(8), dp(16), Math.max(dp(20), safeBottomInset + dp(18)));
            }
            return insets;
        });

        registerPredictiveBackHandler();
        setContentView(root);
    }

    private void registerPredictiveBackHandler() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            backCallback = () -> {
                if (isSheetOpen) {
                    setBottomSheetOpen(false);
                }
            };
        }
    }

    private void updateBackCallbackState() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && backCallback != null) {
            OnBackInvokedDispatcher dispatcher = getOnBackInvokedDispatcher();
            if (dispatcher != null) {
                try {
                    dispatcher.unregisterOnBackInvokedCallback(backCallback);
                } catch (Throwable ignored) {}
                if (isSheetOpen) {
                    dispatcher.registerOnBackInvokedCallback(
                        OnBackInvokedDispatcher.PRIORITY_DEFAULT, backCallback);
                }
            }
        }
    }

    @Override
    @SuppressWarnings("deprecation")
    public void onBackPressed() {
        if (isSheetOpen) {
            setBottomSheetOpen(false);
            return;
        }
        super.onBackPressed();
    }

    @SuppressWarnings("deprecation")
    private void configureImmersiveHighRefreshWindow() {
        Window window = getWindow();
        if (window == null) return;

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            WindowManager.LayoutParams lp = window.getAttributes();
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
            window.setAttributes(lp);
        }

        try {
            Display display = getWindowManager().getDefaultDisplay();
            if (display != null) {
                Display.Mode current = display.getMode();
                Display.Mode[] modes = display.getSupportedModes();
                Display.Mode best = current;
                for (Display.Mode m : modes) {
                    if (m.getPhysicalWidth() == current.getPhysicalWidth()
                        && m.getPhysicalHeight() == current.getPhysicalHeight()
                        && m.getRefreshRate() > best.getRefreshRate()) {
                        best = m;
                    }
                }
                if (best.getModeId() != current.getModeId()) {
                    WindowManager.LayoutParams lp = window.getAttributes();
                    lp.preferredDisplayModeId = best.getModeId();
                    window.setAttributes(lp);
                }
            }
        } catch (Throwable ignored) {}
    }

    private FrameLayout.LayoutParams createBottomSheetLayoutParams(Configuration config) {
        DisplayMetrics dm = getResources().getDisplayMetrics();
        boolean landscape = (config.orientation == Configuration.ORIENTATION_LANDSCAPE);
        int width = landscape ? Math.min(dp(460), dm.widthPixels - dp(32)) : ViewGroup.LayoutParams.MATCH_PARENT;
        int sheetH = landscape
            ? Math.min(dp(420), (int) (dm.heightPixels * 0.80f))
            : Math.max(dp(370), Math.min(dp(500), (int) (dm.heightPixels * 0.53f)));
        return new FrameLayout.LayoutParams(width, sheetH, Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL);
    }

    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        cfg.isPortrait = (newConfig.orientation != Configuration.ORIENTATION_LANDSCAPE);
        if (bottomSheet != null) {
            bottomSheet.setLayoutParams(createBottomSheetLayoutParams(newConfig));
        }
    }

    // =========================================================================
    // 1. TOP MINIMAL ANDROID STATUS OVERLAY
    // =========================================================================

    private View buildTopOverlay() {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);

        // Left: Single-line Status Pill (tap to toggle GPU name detail)
        LinearLayout statusPill = new LinearLayout(this);
        statusPill.setOrientation(LinearLayout.HORIZONTAL);
        statusPill.setGravity(Gravity.CENTER_VERTICAL);
        statusPill.setMinimumHeight(dp(38));
        statusPill.setPadding(dp(12), 0, dp(14), 0);
        statusPill.setElevation(dp(4));
        statusPill.setBackground(ripplePill(M3_HUD_SURFACE, 0, dp(19)));

        statusDotView = new View(this);
        statusDotView.setBackground(circleDrawable(M3_DOT_GPU));
        LinearLayout.LayoutParams dotLp = new LinearLayout.LayoutParams(dp(8), dp(8));
        dotLp.rightMargin = dp(8);
        statusPill.addView(statusDotView, dotLp);

        tvStatusText = new TextView(this);
        tvStatusText.setText("60 FPS · 2.3k");
        tvStatusText.setTextColor(M3_ON_SURFACE);
        tvStatusText.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f);
        tvStatusText.setTypeface(TF_MEDIUM);
        tvStatusText.setSingleLine(true);
        tvStatusText.setMaxLines(1);
        tvStatusText.setEllipsize(TextUtils.TruncateAt.END);
        tvStatusText.setMaxWidth(dp(185));
        statusPill.addView(tvStatusText);

        statusPill.setOnClickListener(v -> {
            hapticTick(v);
            showDetailedStats = !showDetailedStats;
            tvStatusText.setMaxWidth(showDetailedStats ? dp(215) : dp(165));
            refreshStatusText();
        });
        row.addView(statusPill);

        // Flexible spacer
        View spacer = new View(this);
        row.addView(spacer, new LinearLayout.LayoutParams(0, 1, 1.0f));

        // Right Cluster: [2D/3D Pill]  [Tilt Circle]  [Reset Circle]
        btnTopDimension = new LinearLayout(this);
        btnTopDimension.setOrientation(LinearLayout.HORIZONTAL);
        btnTopDimension.setGravity(Gravity.CENTER);
        btnTopDimension.setMinimumHeight(dp(38));
        btnTopDimension.setPadding(dp(12), 0, dp(13), 0);
        btnTopDimension.setElevation(dp(4));
        btnTopDimension.setBackground(ripplePill(M3_HUD_SURFACE, 0, dp(19)));

        iconTopDimension = new MaterialIconDrawable(MaterialIconDrawable.ICON_MODE_2D, M3_ON_SURFACE, dp(17));
        ImageView ivDim = new ImageView(this);
        ivDim.setImageDrawable(iconTopDimension);
        LinearLayout.LayoutParams ivDimLp = new LinearLayout.LayoutParams(dp(17), dp(17));
        ivDimLp.rightMargin = dp(5);
        btnTopDimension.addView(ivDim, ivDimLp);

        tvTopDimension = new TextView(this);
        tvTopDimension.setText("2D");
        tvTopDimension.setTextColor(M3_ON_SURFACE);
        tvTopDimension.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f);
        tvTopDimension.setTypeface(TF_MEDIUM);
        tvTopDimension.setSingleLine(true);
        btnTopDimension.addView(tvTopDimension);

        btnTopDimension.setOnClickListener(v -> {
            hapticTick(v);
            applyDimensionMode(!cfg.is2D);
        });
        row.addView(btnTopDimension, new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, dp(38)));

        // Tilt Sensor Circle Toggle
        iconTopTilt = new MaterialIconDrawable(MaterialIconDrawable.ICON_TILT, M3_ON_SURFACE, dp(20));
        btnTopTilt = makeCircleIconButton(iconTopTilt, dp(38), M3_HUD_SURFACE, v -> {
            hapticTick(v);
            applyTiltGravity(!cfg.tiltGravityEnabled);
        });
        LinearLayout.LayoutParams tiltLp = new LinearLayout.LayoutParams(dp(38), dp(38));
        tiltLp.leftMargin = dp(8);
        row.addView(btnTopTilt, tiltLp);

        // Reset Circle Button
        MaterialIconDrawable iconReset = new MaterialIconDrawable(MaterialIconDrawable.ICON_RESET, M3_ON_SURFACE, dp(20));
        FrameLayout btnReset = makeCircleIconButton(iconReset, dp(38), M3_HUD_SURFACE, v -> {
            hapticClick(v);
            cfg.reloadSceneRequested.set(true);
        });
        LinearLayout.LayoutParams resetLp = new LinearLayout.LayoutParams(dp(38), dp(38));
        resetLp.leftMargin = dp(8);
        row.addView(btnReset, resetLp);

        return row;
    }

    // =========================================================================
    // 2. BOTTOM MATERIAL 3 THUMB DOCK + SQUIRCLE PLAY/PAUSE FAB
    // =========================================================================

    private LinearLayout buildBottomThumbDock() {
        LinearLayout dock = new LinearLayout(this);
        dock.setOrientation(LinearLayout.HORIZONTAL);
        dock.setGravity(Gravity.CENTER_VERTICAL);
        dock.setClipToPadding(false);

        // Left/Center: M3 Floating Segmented Pill [ Pull | Push | Orbit ]
        LinearLayout segContainer = new LinearLayout(this);
        segContainer.setOrientation(LinearLayout.HORIZONTAL);
        segContainer.setGravity(Gravity.CENTER_VERTICAL);
        segContainer.setPadding(dp(4), dp(4), dp(4), dp(4));
        segContainer.setElevation(dp(8));
        segContainer.setBackground(roundRect(M3_DOCK_SURFACE, M3_DIVIDER, dp(27)));

        segPull = new SegmentView(MaterialIconDrawable.ICON_PULL, "Pull", true, () -> selectTouchTool(SimulationConfig.TOOL_ATTRACT));
        segPush = new SegmentView(MaterialIconDrawable.ICON_PUSH, "Push", false, () -> selectTouchTool(SimulationConfig.TOOL_REPEL));
        segOrbit = new SegmentView(MaterialIconDrawable.ICON_ORBIT, "Orbit", false, () -> selectTouchTool(SimulationConfig.TOOL_CAMERA));

        segContainer.addView(segPull, new LinearLayout.LayoutParams(0, dp(46), 1.0f));
        segContainer.addView(segPush, new LinearLayout.LayoutParams(0, dp(46), 1.0f));
        segContainer.addView(segOrbit, new LinearLayout.LayoutParams(0, dp(46), 1.0f));

        dock.addView(segContainer, new LinearLayout.LayoutParams(0, dp(54), 1.0f));

        // Tune / Controls Bottom Sheet Button (52dp Tonal Circle)
        MaterialIconDrawable iconTune = new MaterialIconDrawable(MaterialIconDrawable.ICON_TUNE, M3_ON_SURFACE, dp(22));
        FrameLayout btnTune = makeCircleIconButton(iconTune, dp(52), M3_TONAL_BTN, v -> {
            hapticClick(v);
            setBottomSheetOpen(true);
        });
        btnTune.setElevation(dp(8));
        LinearLayout.LayoutParams tuneLp = new LinearLayout.LayoutParams(dp(52), dp(52));
        tuneLp.leftMargin = dp(10);
        dock.addView(btnTune, tuneLp);

        // Primary Material You Squircle Play/Pause FAB (56dp, 18dp corner radius)
        fabPlayPause = new FrameLayout(this);
        fabPlayPause.setElevation(dp(10));
        iconFabPlay = new MaterialIconDrawable(MaterialIconDrawable.ICON_PAUSE, M3_ON_PRIMARY, dp(24));
        ImageView ivFab = new ImageView(this);
        ivFab.setImageDrawable(iconFabPlay);
        fabPlayPause.addView(ivFab, new FrameLayout.LayoutParams(dp(24), dp(24), Gravity.CENTER));
        stylePlayFab(true);

        fabPlayPause.setOnClickListener(v -> {
            hapticClick(v);
            cfg.isPlaying = !cfg.isPlaying;
            stylePlayFab(cfg.isPlaying);
        });
        LinearLayout.LayoutParams fabLp = new LinearLayout.LayoutParams(dp(56), dp(56));
        fabLp.leftMargin = dp(10);
        dock.addView(fabPlayPause, fabLp);

        return dock;
    }

    private void stylePlayFab(boolean playing) {
        if (playing) {
            iconFabPlay.setIconType(MaterialIconDrawable.ICON_PAUSE);
            iconFabPlay.setTintColor(M3_ON_PRIMARY);
            fabPlayPause.setBackground(ripplePill(M3_PRIMARY, 0, dp(18)));
        } else {
            iconFabPlay.setIconType(MaterialIconDrawable.ICON_PLAY);
            iconFabPlay.setTintColor(M3_ON_PRIMARY_CONT);
            fabPlayPause.setBackground(ripplePill(M3_PRIMARY_CONTAINER, 0, dp(18)));
        }
    }

    private void selectTouchTool(int tool) {
        cfg.touchTool = tool;
        segPull.setSelectedState(tool == SimulationConfig.TOOL_ATTRACT);
        segPush.setSelectedState(tool == SimulationConfig.TOOL_REPEL);
        segOrbit.setSelectedState(tool == SimulationConfig.TOOL_CAMERA);
    }

    private void applyDimensionMode(boolean mode2D) {
        if (cfg.is2D == mode2D) return;
        cfg.is2D = mode2D;
        iconTopDimension.setIconType(mode2D ? MaterialIconDrawable.ICON_MODE_2D : MaterialIconDrawable.ICON_MODE_3D);
        tvTopDimension.setText(mode2D ? "2D" : "3D");
        if (chipDim2D != null) chipDim2D.setCheckedState(mode2D);
        if (chipDim3D != null) chipDim3D.setCheckedState(!mode2D);
        cfg.reloadSceneRequested.set(true);
    }

    private void applyTiltGravity(boolean enabled) {
        cfg.tiltGravityEnabled = enabled;
        if (enabled) {
            iconTopTilt.setTintColor(M3_ON_PRIMARY_CONT);
            btnTopTilt.setBackground(ripplePill(M3_PRIMARY_CONTAINER, 0, dp(19)));
        } else {
            iconTopTilt.setTintColor(M3_ON_SURFACE);
            btnTopTilt.setBackground(ripplePill(M3_HUD_SURFACE, 0, dp(19)));
        }
        if (swTiltGravity != null && swTiltGravity.isChecked() != enabled) {
            swTiltGravity.setChecked(enabled, true);
        }
        updateSensorRegistration();
    }

    // =========================================================================
    // 3. INTERACTIVE MATERIAL 3 BOTTOM SHEET
    // =========================================================================

    private LinearLayout buildMaterialBottomSheet() {
        LinearLayout sheet = new LinearLayout(this);
        sheet.setOrientation(LinearLayout.VERTICAL);
        sheet.setElevation(dp(24));
        sheet.setClickable(true);

        GradientDrawable bg = new GradientDrawable();
        bg.setColor(M3_SHEET_SURFACE);
        float r = dp(28);
        bg.setCornerRadii(new float[]{ r, r, r, r, 0f, 0f, 0f, 0f });
        bg.setStroke(dp(1), 0x22CBE6FF);
        sheet.setBackground(bg);
        sheet.setVisibility(View.GONE);

        // Drag Handle + Header Zone (supports swipe-down to dismiss without blocking child taps!)
        DraggableSheetHeader dragHeaderZone = new DraggableSheetHeader(sheet);
        dragHeaderZone.setOrientation(LinearLayout.VERTICAL);
        dragHeaderZone.setPadding(dp(20), dp(10), dp(16), dp(6));

        // M3 Drag Handle Pill (36dp x 4dp)
        View handle = new View(this);
        handle.setBackground(roundRect(0xFF64748B, 0, dp(2)));
        LinearLayout.LayoutParams handleLp = new LinearLayout.LayoutParams(dp(36), dp(4));
        handleLp.gravity = Gravity.CENTER_HORIZONTAL;
        handleLp.bottomMargin = dp(10);
        dragHeaderZone.addView(handle, handleLp);

        // Header Row: Title + [Step] [Fit] [Collapse]
        LinearLayout headerRow = new LinearLayout(this);
        headerRow.setOrientation(LinearLayout.HORIZONTAL);
        headerRow.setGravity(Gravity.CENTER_VERTICAL);

        TextView sheetTitle = new TextView(this);
        sheetTitle.setText("Controls");
        sheetTitle.setTextColor(M3_ON_SURFACE);
        sheetTitle.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f);
        sheetTitle.setTypeface(TF_MEDIUM);
        headerRow.addView(sheetTitle, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));

        // Step 1 Frame Tonal Pill
        LinearLayout btnStep = makeCompactActionChip(MaterialIconDrawable.ICON_STEP, "Step", v -> {
            hapticTick(v);
            cfg.isPlaying = false;
            stylePlayFab(false);
            cfg.stepOnceRequested.set(true);
        });
        headerRow.addView(btnStep);

        // Recenter Camera Tonal Pill
        LinearLayout btnFit = makeCompactActionChip(MaterialIconDrawable.ICON_CENTER, "Fit", v -> {
            hapticTick(v);
            cfg.resetCameraRequested.set(true);
        });
        LinearLayout.LayoutParams fitLp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, dp(34));
        fitLp.leftMargin = dp(8);
        headerRow.addView(btnFit, fitLp);

        // Collapse Chevron Circle
        MaterialIconDrawable iconDown = new MaterialIconDrawable(MaterialIconDrawable.ICON_CHEVRON_DOWN, M3_ON_SURFACE, dp(20));
        FrameLayout btnClose = makeCircleIconButton(iconDown, dp(36), M3_TONAL_BTN, v -> {
            hapticTick(v);
            setBottomSheetOpen(false);
        });
        LinearLayout.LayoutParams closeLp = new LinearLayout.LayoutParams(dp(36), dp(36));
        closeLp.leftMargin = dp(8);
        headerRow.addView(btnClose, closeLp);

        dragHeaderZone.addView(headerRow);
        sheet.addView(dragHeaderZone);

        // Material 3 Primary Underline Tab Bar
        LinearLayout tabRow = new LinearLayout(this);
        tabRow.setOrientation(LinearLayout.HORIZONTAL);
        tabRow.setPadding(dp(12), 0, dp(12), 0);

        tabScene = new TabItemView("Scene", true, () -> selectSheetTab(0));
        tabPhysics = new TabItemView("Physics", false, () -> selectSheetTab(1));
        tabAppearance = new TabItemView("Appearance", false, () -> selectSheetTab(2));

        tabRow.addView(tabScene, new LinearLayout.LayoutParams(0, dp(44), 1.0f));
        tabRow.addView(tabPhysics, new LinearLayout.LayoutParams(0, dp(44), 1.0f));
        tabRow.addView(tabAppearance, new LinearLayout.LayoutParams(0, dp(44), 1.0f));
        sheet.addView(tabRow);

        View divider = new View(this);
        divider.setBackgroundColor(M3_DIVIDER);
        sheet.addView(divider, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1)));

        // Scrollable Sheet Content
        sheetScrollView = new ScrollView(this);
        sheetScrollView.setVerticalScrollBarEnabled(false);
        sheetScrollView.setClipToPadding(false);
        sheetScrollView.setPadding(dp(16), dp(12), dp(16), dp(28));

        FrameLayout host = new FrameLayout(this);
        paneScene = buildScenePane();
        panePhysics = buildPhysicsPane();
        paneAppearance = buildAppearancePane();

        panePhysics.setVisibility(View.GONE);
        paneAppearance.setVisibility(View.GONE);

        host.addView(paneScene);
        host.addView(panePhysics);
        host.addView(paneAppearance);
        sheetScrollView.addView(host);

        sheet.addView(sheetScrollView, new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.0f));

        return sheet;
    }

    private final class DraggableSheetHeader extends LinearLayout {
        private final View targetSheet;
        private float downRawY = 0f;
        private boolean isDragging = false;
        private VelocityTracker vt;

        DraggableSheetHeader(View targetSheet) {
            super(MainActivity.this);
            this.targetSheet = targetSheet;
        }

        @Override
        public boolean onInterceptTouchEvent(MotionEvent ev) {
            int action = ev.getActionMasked();
            if (action == MotionEvent.ACTION_DOWN) {
                downRawY = ev.getRawY();
                isDragging = false;
                if (vt == null) vt = VelocityTracker.obtain();
                else vt.clear();
                vt.addMovement(ev);
                return false;
            }
            if (action == MotionEvent.ACTION_MOVE) {
                if (vt != null) vt.addMovement(ev);
                if (Math.abs(ev.getRawY() - downRawY) > dp(8)) {
                    isDragging = true;
                    return true;
                }
            }
            return false;
        }

        @Override
        public boolean onTouchEvent(MotionEvent ev) {
            int action = ev.getActionMasked();
            if (action == MotionEvent.ACTION_DOWN) {
                downRawY = ev.getRawY();
                isDragging = false;
                if (vt == null) vt = VelocityTracker.obtain();
                else vt.clear();
                vt.addMovement(ev);
                return true;
            }
            if (action == MotionEvent.ACTION_MOVE) {
                if (vt != null) vt.addMovement(ev);
                float dy = ev.getRawY() - downRawY;
                if (dy > dp(4)) isDragging = true;
                if (isDragging) {
                    targetSheet.setTranslationY(Math.max(0f, dy));
                }
                return true;
            }
            if (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL) {
                float vy = 0f;
                if (vt != null) {
                    vt.addMovement(ev);
                    vt.computeCurrentVelocity(1000);
                    vy = vt.getYVelocity();
                    vt.recycle();
                    vt = null;
                }
                float trans = targetSheet.getTranslationY();
                if (trans > dp(85) || vy > 850f) {
                    setBottomSheetOpen(false);
                } else if (trans > 0f) {
                    targetSheet.animate()
                        .translationY(0f)
                        .setDuration(180)
                        .setInterpolator(new DecelerateInterpolator())
                        .start();
                }
                isDragging = false;
                return true;
            }
            return super.onTouchEvent(ev);
        }
    }

    private void setBottomSheetOpen(boolean open) {
        if (this.isSheetOpen == open) return;
        this.isSheetOpen = open;
        updateBackCallbackState();

        int h = bottomSheet.getHeight();
        if (h <= 0) {
            h = dp(420);
        }

        if (open) {
            bottomSheet.setVisibility(View.VISIBLE);
            bottomSheet.setTranslationY(h);
            bottomSheet.animate()
                .translationY(0f)
                .setDuration(250)
                .setInterpolator(new DecelerateInterpolator(1.6f))
                .start();

            bottomDock.animate()
                .alpha(0f)
                .translationY(dp(24))
                .setDuration(160)
                .withEndAction(() -> {
                    if (isSheetOpen) bottomDock.setVisibility(View.INVISIBLE);
                })
                .start();
        } else {
            bottomDock.setVisibility(View.VISIBLE);
            bottomDock.animate()
                .alpha(1f)
                .translationY(0f)
                .setDuration(210)
                .setInterpolator(new DecelerateInterpolator())
                .start();

            bottomSheet.animate()
                .translationY(h)
                .setDuration(220)
                .setInterpolator(new DecelerateInterpolator(1.4f))
                .withEndAction(() -> {
                    if (!isSheetOpen) {
                        bottomSheet.setVisibility(View.GONE);
                        bottomSheet.setTranslationY(0f);
                    }
                })
                .start();
        }
    }

    private void selectSheetTab(int idx) {
        tabScene.setActive(idx == 0);
        tabPhysics.setActive(idx == 1);
        tabAppearance.setActive(idx == 2);

        paneScene.setVisibility(idx == 0 ? View.VISIBLE : View.GONE);
        panePhysics.setVisibility(idx == 1 ? View.VISIBLE : View.GONE);
        paneAppearance.setVisibility(idx == 2 ? View.VISIBLE : View.GONE);
        if (sheetScrollView != null) {
            sheetScrollView.scrollTo(0, 0);
        }
    }

    // =========================================================================
    // 4. SHEET PANE 1: SCENE
    // =========================================================================

    private LinearLayout buildScenePane() {
        LinearLayout col = makePaneColumn();

        // Section: Scene Preset Chips
        addSectionHeader(col, "Scene Preset");
        presetChips = new FilterChipView[SimulationConfig.PRESET_NAMES.length];
        LinearLayout presetRow = makeChipStrip(col);
        for (int i = 0; i < SimulationConfig.PRESET_NAMES.length; i++) {
            final int idx = i;
            presetChips[i] = new FilterChipView(SimulationConfig.PRESET_NAMES[i], i == 0, () -> {
                selectPresetChip(idx);
            });
            presetRow.addView(presetChips[i], chipMargin(i));
        }

        // Section: Simulation Quality & Mode Card
        LinearLayout cardSim = makeM3Card();
        addCardTitle(cardSim, "Simulation Space & Density");

        LinearLayout dimRow = new LinearLayout(this);
        dimRow.setOrientation(LinearLayout.HORIZONTAL);
        chipDim2D = new FilterChipView("2D Planar", true, () -> applyDimensionMode(true));
        chipDim3D = new FilterChipView("3D Volumetric", false, () -> applyDimensionMode(false));
        dimRow.addView(chipDim2D, chipMargin(0));
        dimRow.addView(chipDim3D, chipMargin(1));
        cardSim.addView(dimRow);

        TextView qualLbl = makeSubtleLabel("Particle Density Tier");
        qualLbl.setPadding(0, dp(12), 0, dp(6));
        cardSim.addView(qualLbl);

        qualityChips = new FilterChipView[SimulationConfig.QUALITY_NAMES.length];
        LinearLayout qualRow = makeChipStrip(cardSim);
        for (int i = 0; i < SimulationConfig.QUALITY_NAMES.length; i++) {
            final int idx = i;
            qualityChips[i] = new FilterChipView(SimulationConfig.QUALITY_NAMES[i], i == 0, () -> {
                cfg.quality = idx;
                for (int k = 0; k < qualityChips.length; k++) {
                    qualityChips[k].setCheckedState(k == idx);
                }
                cfg.reloadSceneRequested.set(true);
            });
            qualRow.addView(qualityChips[i], chipMargin(i));
        }

        tvKernelInfo = makeSubtleLabel("Smoothing radius h: 0.057 m");
        tvKernelInfo.setPadding(0, dp(8), 0, 0);
        cardSim.addView(tvKernelInfo);
        col.addView(cardSim, cardTopGap());

        // Section: Time Scale Card
        LinearLayout cardSpeed = makeM3Card();
        addCardTitle(cardSpeed, "Time Scale");

        LinearLayout speedChips = new LinearLayout(this);
        speedChips.setOrientation(LinearLayout.HORIZONTAL);
        chipSpeedSlow = new FilterChipView("0.25× Slow", false, () -> applySpeedPreset(0.25f));
        chipSpeedSmooth = new FilterChipView("0.5× Smooth", false, () -> applySpeedPreset(0.50f));
        chipSpeedReal = new FilterChipView("1.0× Real", true, () -> applySpeedPreset(1.00f));
        speedChips.addView(chipSpeedSlow, chipMargin(0));
        speedChips.addView(chipSpeedSmooth, chipMargin(1));
        speedChips.addView(chipSpeedReal, chipMargin(2));
        cardSpeed.addView(speedChips);

        tvSpeedBadge = new TextView(this);
        sbSpeed = addM3Slider(cardSpeed, "Playback Speed", "1.00×", tvSpeedBadge, 5, 200, 100, val -> {
            float s = val / 100.0f;
            cfg.simSpeed = s;
            tvSpeedBadge.setText(String.format(Locale.US, "%.2f×", s));
            chipSpeedSlow.setCheckedState(Math.abs(s - 0.25f) < 0.02f);
            chipSpeedSmooth.setCheckedState(Math.abs(s - 0.50f) < 0.02f);
            chipSpeedReal.setCheckedState(Math.abs(s - 1.00f) < 0.02f);
        });
        col.addView(cardSpeed, cardTopGap());

        // Section: Sphere Collider Card (with collapsible sliders!)
        LinearLayout cardObs = makeM3Card();
        obstacleSlidersWrap = new LinearLayout(this);
        obstacleSlidersWrap.setOrientation(LinearLayout.VERTICAL);
        obstacleSlidersWrap.setVisibility(View.GONE);

        swObstacle = addSwitchRow(cardObs, "Sphere Collider", "Solid interactive obstacle inside tank", false, checked -> {
            cfg.obstacleEnabled = checked;
            obstacleSlidersWrap.setVisibility(checked ? View.VISIBLE : View.GONE);
        });

        TextView tvOx = new TextView(this);
        addM3Slider(obstacleSlidersWrap, "Horizontal X", "0.00 m", tvOx, 0, 110, 55, val -> {
            float x = (val - 55) / 100.0f;
            cfg.obstacleX = x;
            tvOx.setText(String.format(Locale.US, "%.2f m", x));
        });

        TextView tvOy = new TextView(this);
        addM3Slider(obstacleSlidersWrap, "Vertical Y", "0.45 m", tvOy, 15, 140, 45, val -> {
            float y = val / 100.0f;
            cfg.obstacleY = y;
            tvOy.setText(String.format(Locale.US, "%.2f m", y));
        });

        TextView tvOr = new TextView(this);
        addM3Slider(obstacleSlidersWrap, "Radius", "0.20 m", tvOr, 8, 40, 20, val -> {
            float r = val / 100.0f;
            cfg.obstacleR = r;
            tvOr.setText(String.format(Locale.US, "%.2f m", r));
        });

        cardObs.addView(obstacleSlidersWrap);
        col.addView(cardObs, cardTopGap());

        return col;
    }

    private void selectPresetChip(int idx) {
        cfg.preset = idx;
        for (int i = 0; i < presetChips.length; i++) {
            presetChips[i].setCheckedState(i == idx);
        }
        boolean hasObs = (idx == SimulationConfig.PRESET_OBSTACLE);
        cfg.obstacleEnabled = hasObs;
        if (swObstacle != null) swObstacle.setChecked(hasObs, true);
        if (obstacleSlidersWrap != null) obstacleSlidersWrap.setVisibility(hasObs ? View.VISIBLE : View.GONE);

        if (idx == SimulationConfig.PRESET_ZERO_G) {
            syncGravitySlider(0.0f);
        } else if (cfg.gravityMag < 0.1f) {
            syncGravitySlider(9.81f);
        }
        cfg.reloadSceneRequested.set(true);
    }

    private void applySpeedPreset(float s) {
        cfg.simSpeed = s;
        if (sbSpeed != null) sbSpeed.setProgress(Math.round(s * 100f) - 5);
        if (tvSpeedBadge != null) tvSpeedBadge.setText(String.format(Locale.US, "%.2f×", s));
        chipSpeedSlow.setCheckedState(Math.abs(s - 0.25f) < 0.02f);
        chipSpeedSmooth.setCheckedState(Math.abs(s - 0.50f) < 0.02f);
        chipSpeedReal.setCheckedState(Math.abs(s - 1.00f) < 0.02f);
    }

    private void syncGravitySlider(float g) {
        cfg.gravityMag = g;
        if (sbGravity != null) sbGravity.setProgress(Math.round(g * 10f));
        if (tvGravityBadge != null) tvGravityBadge.setText(String.format(Locale.US, "%.1f m/s²", g));
    }

    // =========================================================================
    // 5. SHEET PANE 2: PHYSICS
    // =========================================================================

    private LinearLayout buildPhysicsPane() {
        LinearLayout col = makePaneColumn();

        // Card 1: Motion Sensor Tilt Gravity
        LinearLayout cardTilt = makeM3Card();
        swTiltGravity = addSwitchRow(cardTilt,
            "Phone Tilt Gravity",
            "Tilt your phone to pour and slosh the liquid",
            cfg.tiltGravityEnabled,
            this::applyTiltGravity);
        col.addView(cardTilt);

        // Card 2: Fluid Solver Parameters
        LinearLayout cardDyn = makeM3Card();
        addCardTitle(cardDyn, "Fluid Solver Parameters");

        TextView tvVort = new TextView(this);
        addM3Slider(cardDyn, "Vorticity Swirl", "0.045", tvVort, 0, 180, 45, val -> {
            float v = val / 1000.0f;
            cfg.vorticity = v;
            tvVort.setText(String.format(Locale.US, "%.3f", v));
        });

        TextView tvCoh = new TextView(this);
        addM3Slider(cardDyn, "Surface Cohesion", "0.030", tvCoh, 0, 150, 30, val -> {
            float c = val / 1000.0f;
            cfg.surfaceTension = c;
            tvCoh.setText(String.format(Locale.US, "%.3f", c));
        });

        TextView tvVisc = new TextView(this);
        addM3Slider(cardDyn, "Viscosity", "0.035", tvVisc, 0, 300, 35, val -> {
            float v = val / 1000.0f;
            cfg.viscosity = v;
            tvVisc.setText(String.format(Locale.US, "%.3f", v));
        });

        tvGravityBadge = new TextView(this);
        sbGravity = addM3Slider(cardDyn, "Gravity Strength", "9.8 m/s²", tvGravityBadge, 0, 250, 98, val -> {
            float g = val / 10.0f;
            cfg.gravityMag = g;
            tvGravityBadge.setText(String.format(Locale.US, "%.1f m/s²", g));
        });

        col.addView(cardDyn, cardTopGap());
        return col;
    }

    // =========================================================================
    // 6. SHEET PANE 3: APPEARANCE (OPTICAL MATERIALS)
    // =========================================================================

    private LinearLayout buildAppearancePane() {
        LinearLayout col = makePaneColumn();

        // Liquid Material Swatches
        addSectionHeader(col, "Liquid Material");
        LinearLayout swatchRow = makeChipStrip(col);
        materialSwatchChips = new SwatchChipView[SimulationConfig.MATERIALS.length];
        for (int i = 0; i < SimulationConfig.MATERIALS.length; i++) {
            final int idx = i;
            SimulationConfig.MaterialPreset m = SimulationConfig.MATERIALS[i];
            materialSwatchChips[i] = new SwatchChipView(m.name, m.shallowColor, m.deepColor, i == 0, () -> {
                for (int k = 0; k < materialSwatchChips.length; k++) {
                    materialSwatchChips[k].setSelectedState(k == idx);
                }
                cfg.applyMaterialPreset(idx);
                syncOpticalSlidersFromConfig();
            });
            swatchRow.addView(materialSwatchChips[i], chipMargin(i));
        }

        // Card 1: Shading Mode
        LinearLayout cardShade = makeM3Card();
        addCardTitle(cardShade, "Shading Mode");

        String[] shadeNames = { "Optical Liquid", "Velocity", "Pressure", "Density" };
        colorModeChips = new FilterChipView[shadeNames.length];
        LinearLayout shadeStrip = makeChipStrip(cardShade);
        for (int i = 0; i < shadeNames.length; i++) {
            final int idx = i;
            colorModeChips[i] = new FilterChipView(shadeNames[i], i == 0, () -> {
                for (int k = 0; k < colorModeChips.length; k++) {
                    colorModeChips[k].setCheckedState(k == idx);
                }
                cfg.colorMode = (idx == 0) ? 3 : (idx - 1);
                if (colorRampWrap != null) {
                    colorRampWrap.setVisibility(idx == 0 ? View.GONE : View.VISIBLE);
                }
            });
            shadeStrip.addView(colorModeChips[i], chipMargin(i));
        }

        colorRampWrap = new LinearLayout(this);
        colorRampWrap.setOrientation(LinearLayout.VERTICAL);
        colorRampWrap.setVisibility(View.GONE);
        TextView rampLbl = makeSubtleLabel("Color Gradient");
        rampLbl.setPadding(0, dp(10), 0, dp(6));
        colorRampWrap.addView(rampLbl);

        String[] ramps = { "Ocean", "Turbo", "Viridis", "Magma" };
        colorRampChips = new FilterChipView[ramps.length];
        LinearLayout rampStrip = makeChipStrip(colorRampWrap);
        for (int i = 0; i < ramps.length; i++) {
            final int idx = i;
            colorRampChips[i] = new FilterChipView(ramps[i], i == 0, () -> {
                cfg.colorMap = idx;
                for (int k = 0; k < colorRampChips.length; k++) {
                    colorRampChips[k].setCheckedState(k == idx);
                }
            });
            rampStrip.addView(colorRampChips[i], chipMargin(i));
        }
        cardShade.addView(colorRampWrap);

        View div = new View(this);
        div.setBackgroundColor(M3_DIVIDER);
        LinearLayout.LayoutParams divLp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1));
        divLp.setMargins(0, dp(12), 0, dp(4));
        cardShade.addView(div, divLp);

        addSwitchRow(cardShade, "Specular & Sky Reflections", "Fresnel rim lighting and sun highlights", true, checked -> {
            cfg.litSpheres = checked;
        });
        col.addView(cardShade, cardTopGap());

        // Card 2: Surface Texture
        LinearLayout cardTex = makeM3Card();
        addCardTitle(cardTex, "Surface Texture");

        String[] texNames = { "Ripples", "Caustics", "Foam", "Grid", "Smooth" };
        final int[] texIds = { 1, 2, 3, 4, 0 };
        texModeChips = new FilterChipView[texNames.length];
        LinearLayout texStrip = makeChipStrip(cardTex);
        for (int i = 0; i < texNames.length; i++) {
            final int idx = i;
            texModeChips[i] = new FilterChipView(texNames[i], i == 0, () -> {
                cfg.texMode = texIds[idx];
                for (int k = 0; k < texModeChips.length; k++) {
                    texModeChips[k].setCheckedState(k == idx);
                }
            });
            texStrip.addView(texModeChips[i], chipMargin(i));
        }

        TextView tvTexStr = new TextView(this);
        addM3Slider(cardTex, "Pattern Depth", "0.28", tvTexStr, 0, 100, 28, val -> {
            float v = val / 100.0f;
            cfg.texStrength = v;
            tvTexStr.setText(String.format(Locale.US, "%.2f", v));
        });

        tvFoamVal = new TextView(this);
        sbFoam = addM3Slider(cardTex, "Crest Whitecaps", "0.35", tvFoamVal, 0, 100, 35, val -> {
            if (suppressUiSync) return;
            float v = val / 100.0f;
            cfg.foamAmount = v;
            tvFoamVal.setText(String.format(Locale.US, "%.2f", v));
        });
        col.addView(cardTex, cardTopGap());

        // Card 3: Optical Light Behavior
        LinearLayout cardOpt = makeM3Card();
        addCardTitle(cardOpt, "Optical Properties");

        tvIorVal = new TextView(this);
        sbIor = addM3Slider(cardOpt, "Fresnel IOR", "1.333", tvIorVal, 100, 250, 133, val -> {
            if (suppressUiSync) return;
            float v = val / 100.0f;
            cfg.ior = v;
            tvIorVal.setText(String.format(Locale.US, "%.3f", v));
        });

        tvRoughVal = new TextView(this);
        sbRough = addM3Slider(cardOpt, "Surface Roughness", "0.045", tvRoughVal, 15, 800, 45, val -> {
            if (suppressUiSync) return;
            float v = val / 1000.0f;
            cfg.roughness = v;
            tvRoughVal.setText(String.format(Locale.US, "%.3f", v));
        });

        tvSpecVal = new TextView(this);
        sbSpec = addM3Slider(cardOpt, "Sun Specular", "1.35", tvSpecVal, 0, 300, 135, val -> {
            if (suppressUiSync) return;
            float v = val / 100.0f;
            cfg.specular = v;
            tvSpecVal.setText(String.format(Locale.US, "%.2f", v));
        });

        tvRefrVal = new TextView(this);
        sbRefr = addM3Slider(cardOpt, "Subsurface Glow", "0.42", tvRefrVal, 0, 120, 42, val -> {
            if (suppressUiSync) return;
            float v = val / 100.0f;
            cfg.refraction = v;
            tvRefrVal.setText(String.format(Locale.US, "%.2f", v));
        });

        tvMetalVal = new TextView(this);
        sbMetal = addM3Slider(cardOpt, "Metallic", "0.00", tvMetalVal, 0, 100, 0, val -> {
            if (suppressUiSync) return;
            float v = val / 100.0f;
            cfg.metallic = v;
            tvMetalVal.setText(String.format(Locale.US, "%.2f", v));
        });

        col.addView(cardOpt, cardTopGap());
        return col;
    }

    private void syncOpticalSlidersFromConfig() {
        suppressUiSync = true;
        try {
            if (colorModeChips != null) {
                for (int i = 0; i < colorModeChips.length; i++) {
                    colorModeChips[i].setCheckedState(i == 0);
                }
            }
            if (colorRampWrap != null) colorRampWrap.setVisibility(View.GONE);

            if (sbIor != null) sbIor.setProgress(Math.round(cfg.ior * 100f) - 100);
            if (tvIorVal != null) tvIorVal.setText(String.format(Locale.US, "%.3f", cfg.ior));

            if (sbRough != null) sbRough.setProgress(Math.round(cfg.roughness * 1000f) - 15);
            if (tvRoughVal != null) tvRoughVal.setText(String.format(Locale.US, "%.3f", cfg.roughness));

            if (sbSpec != null) sbSpec.setProgress(Math.round(cfg.specular * 100f));
            if (tvSpecVal != null) tvSpecVal.setText(String.format(Locale.US, "%.2f", cfg.specular));

            if (sbRefr != null) sbRefr.setProgress(Math.round(cfg.refraction * 100f));
            if (tvRefrVal != null) tvRefrVal.setText(String.format(Locale.US, "%.2f", cfg.refraction));

            if (sbMetal != null) sbMetal.setProgress(Math.round(cfg.metallic * 100f));
            if (tvMetalVal != null) tvMetalVal.setText(String.format(Locale.US, "%.2f", cfg.metallic));

            if (sbFoam != null) sbFoam.setProgress(Math.round(cfg.foamAmount * 100f));
            if (tvFoamVal != null) tvFoamVal.setText(String.format(Locale.US, "%.2f", cfg.foamAmount));
        } finally {
            suppressUiSync = false;
        }
    }

    // =========================================================================
    // 7. TELEMETRY & SENSOR CALLBACKS
    // =========================================================================

    private static String cleanGpuName(String raw) {
        if (raw == null) return "GPU";
        String s = raw.replace("(TM)", "").replace("(R)", "").replace("Graphics", "").trim();
        while (s.contains("  ")) s = s.replace("  ", " ");
        return s;
    }

    private static String formatParticlesShort(int n) {
        if (n >= 1000) {
            return String.format(Locale.US, "%.1fk", n / 1000.0f);
        }
        return String.valueOf(n);
    }

    private void refreshStatusText() {
        if (tvStatusText == null) return;
        String pts = formatParticlesShort(lastParticles);
        if (showDetailedStats) {
            tvStatusText.setText(String.format(Locale.US, "%s · %d FPS · %s · %.1fms",
                backendLabel, lastFps, pts, lastStepMs));
        } else {
            tvStatusText.setText(String.format(Locale.US, "%d FPS · %s", lastFps, pts));
        }
    }

    @Override
    public void onBackendReady(boolean isGpuCompute, String gpuName, float kernelH, int particles) {
        runOnUiThread(() -> {
            backendLabel = isGpuCompute ? cleanGpuName(gpuName) : "CPU Multi-Core";
            lastParticles = particles;
            if (statusDotView != null) {
                statusDotView.setBackground(circleDrawable(isGpuCompute ? M3_DOT_GPU : M3_DOT_CPU));
            }
            if (tvKernelInfo != null) {
                tvKernelInfo.setText(String.format(Locale.US, "Smoothing radius h: %.3f m", kernelH));
            }
            refreshStatusText();
        });
    }

    @Override
    public void onTelemetry(int fps, int particles, float stepMs, float kernelH) {
        runOnUiThread(() -> {
            lastFps = fps;
            lastParticles = particles;
            lastStepMs = stepMs;
            refreshStatusText();
            if (tvKernelInfo != null) {
                tvKernelInfo.setText(String.format(Locale.US, "Smoothing radius h: %.3f m", kernelH));
            }
        });
    }

    private void updateSensorRegistration() {
        if (sensorManager == null || gravitySensor == null) return;
        sensorManager.unregisterListener(this);
        if (cfg.tiltGravityEnabled) {
            sensorManager.registerListener(this, gravitySensor, SensorManager.SENSOR_DELAY_GAME);
        }
    }

    @Override
    @SuppressWarnings("deprecation")
    public void onSensorChanged(SensorEvent event) {
        if (!cfg.tiltGravityEnabled) return;

        float sx = event.values[0];
        float sy = event.values[1];
        float sz = event.values[2];

        int rot = Surface.ROTATION_0;
        Display d = getWindowManager().getDefaultDisplay();
        if (d != null) rot = d.getRotation();

        float wx, wy;
        switch (rot) {
            case Surface.ROTATION_90:  wx =  sy; wy = -sx; break;
            case Surface.ROTATION_180: wx =  sx; wy =  sy; break;
            case Surface.ROTATION_270: wx = -sy; wy =  sx; break;
            case Surface.ROTATION_0:
            default:                   wx = -sx; wy = -sy; break;
        }
        float wz = -sz;

        float len = (float) Math.sqrt(wx * wx + wy * wy + wz * wz);
        if (len > 0.2f) {
            wx /= len; wy /= len; wz /= len;
            smoothGx = smoothGx * 0.80f + wx * 0.20f;
            smoothGy = smoothGy * 0.80f + wy * 0.20f;
            smoothGz = smoothGz * 0.80f + wz * 0.20f;

            float sLen = (float) Math.sqrt(smoothGx * smoothGx + smoothGy * smoothGy + smoothGz * smoothGz);
            if (sLen > 1e-4f) {
                cfg.tiltDirX = smoothGx / sLen;
                cfg.tiltDirY = smoothGy / sLen;
                cfg.tiltDirZ = smoothGz / sLen;
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    @Override
    protected void onResume() {
        super.onResume();
        glView.onResume();
        updateSensorRegistration();
    }

    @Override
    protected void onPause() {
        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }
        glView.onPause();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        glView.shutdown();
        super.onDestroy();
    }

    // =========================================================================
    // 8. NATIVE MATERIAL DESIGN 3 CUSTOM COMPONENTS & HELPERS
    // =========================================================================

    private interface IntConsumer { void accept(int value); }
    private interface BoolConsumer { void accept(boolean value); }

    private int dp(float v) {
        return Math.round(TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics()));
    }

    private static void hapticTick(View v) {
        if (v != null) v.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK);
    }

    private static void hapticClick(View v) {
        if (v != null) v.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY);
    }

    private static GradientDrawable roundRect(int fill, int strokeColor, int radiusPx) {
        GradientDrawable gd = new GradientDrawable();
        gd.setColor(fill);
        gd.setCornerRadius(radiusPx);
        if (strokeColor != 0) {
            gd.setStroke(2, strokeColor);
        }
        return gd;
    }

    private static GradientDrawable circleDrawable(int fill) {
        GradientDrawable gd = new GradientDrawable();
        gd.setShape(GradientDrawable.OVAL);
        gd.setColor(fill);
        return gd;
    }

    private static RippleDrawable ripplePill(int fill, int strokeColor, int radiusPx) {
        GradientDrawable content = roundRect(fill, strokeColor, radiusPx);
        GradientDrawable mask = roundRect(Color.WHITE, 0, radiusPx);
        return new RippleDrawable(ColorStateList.valueOf(M3_RIPPLE), content, mask);
    }

    private FrameLayout makeCircleIconButton(Drawable icon, int sizePx, int bgFill, View.OnClickListener onClick) {
        FrameLayout btn = new FrameLayout(this);
        btn.setBackground(ripplePill(bgFill, 0, sizePx / 2));
        ImageView iv = new ImageView(this);
        iv.setImageDrawable(icon);
        int iconSz = icon.getIntrinsicWidth() > 0 ? icon.getIntrinsicWidth() : dp(20);
        btn.addView(iv, new FrameLayout.LayoutParams(iconSz, iconSz, Gravity.CENTER));
        btn.setOnClickListener(onClick);
        return btn;
    }

    private LinearLayout makeCompactActionChip(int iconType, String text, View.OnClickListener onClick) {
        LinearLayout chip = new LinearLayout(this);
        chip.setOrientation(LinearLayout.HORIZONTAL);
        chip.setGravity(Gravity.CENTER_VERTICAL);
        chip.setMinimumHeight(dp(34));
        chip.setPadding(dp(10), 0, dp(12), 0);
        chip.setBackground(ripplePill(M3_TONAL_BTN, 0, dp(17)));

        MaterialIconDrawable ic = new MaterialIconDrawable(iconType, M3_ON_SURFACE, dp(16));
        ImageView iv = new ImageView(this);
        iv.setImageDrawable(ic);
        LinearLayout.LayoutParams ivLp = new LinearLayout.LayoutParams(dp(16), dp(16));
        ivLp.rightMargin = dp(5);
        chip.addView(iv, ivLp);

        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextColor(M3_ON_SURFACE);
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f);
        tv.setTypeface(TF_MEDIUM);
        tv.setSingleLine(true);
        chip.addView(tv);

        chip.setOnClickListener(onClick);
        return chip;
    }

    /** Segmented pill item inside the bottom thumb dock */
    private final class SegmentView extends LinearLayout {
        private final MaterialIconDrawable icon;
        private final TextView label;

        SegmentView(int iconType, String text, boolean active, Runnable onSelect) {
            super(MainActivity.this);
            setOrientation(HORIZONTAL);
            setGravity(Gravity.CENTER);
            setPadding(dp(6), 0, dp(8), 0);

            icon = new MaterialIconDrawable(iconType, M3_ON_SURFACE_VAR, dp(17));
            ImageView iv = new ImageView(MainActivity.this);
            iv.setImageDrawable(icon);
            LayoutParams ivLp = new LayoutParams(dp(17), dp(17));
            ivLp.rightMargin = dp(5);
            addView(iv, ivLp);

            label = new TextView(MainActivity.this);
            label.setText(text);
            label.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f);
            label.setTypeface(TF_MEDIUM);
            label.setSingleLine(true);
            addView(label);

            setSelectedState(active);
            setOnClickListener(v -> {
                hapticTick(v);
                onSelect.run();
            });
        }

        void setSelectedState(boolean selected) {
            if (selected) {
                icon.setTintColor(M3_ON_PRIMARY_CONT);
                label.setTextColor(M3_ON_PRIMARY_CONT);
                setBackground(ripplePill(M3_PRIMARY_CONTAINER, 0, dp(23)));
            } else {
                icon.setTintColor(M3_ON_SURFACE_VAR);
                label.setTextColor(M3_ON_SURFACE_VAR);
                setBackground(ripplePill(Color.TRANSPARENT, 0, dp(23)));
            }
        }
    }

    /** Material 3 Primary Tab Item with rounded bottom indicator pill */
    private final class TabItemView extends FrameLayout {
        private final TextView tv;
        private final View indicator;

        TabItemView(String text, boolean active, Runnable onSelect) {
            super(MainActivity.this);
            setBackground(ripplePill(Color.TRANSPARENT, 0, dp(10)));

            tv = new TextView(MainActivity.this);
            tv.setText(text);
            tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f);
            tv.setTypeface(TF_MEDIUM);
            tv.setGravity(Gravity.CENTER);
            addView(tv, new LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.CENTER));

            indicator = new View(MainActivity.this);
            GradientDrawable indBg = new GradientDrawable();
            indBg.setColor(M3_PRIMARY);
            float r = dp(3);
            indBg.setCornerRadii(new float[]{ r, r, r, r, 0f, 0f, 0f, 0f });
            indicator.setBackground(indBg);
            addView(indicator, new LayoutParams(dp(44), dp(3), Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL));

            setActive(active);
            setOnClickListener(v -> {
                hapticTick(v);
                onSelect.run();
            });
        }

        void setActive(boolean active) {
            tv.setTextColor(active ? M3_PRIMARY : M3_ON_SURFACE_VAR);
            indicator.setVisibility(active ? VISIBLE : INVISIBLE);
        }
    }

    /** Authentic Material 3 FilterChip (8dp corner radius + leading checkmark when selected) */
    private final class FilterChipView extends LinearLayout {
        private final ImageView checkView;
        private final TextView tv;

        FilterChipView(String text, boolean checked, Runnable onSelect) {
            super(MainActivity.this);
            setOrientation(HORIZONTAL);
            setGravity(Gravity.CENTER_VERTICAL);
            setMinimumHeight(dp(36));

            MaterialIconDrawable checkIcon = new MaterialIconDrawable(MaterialIconDrawable.ICON_CHECK, M3_ON_PRIMARY_CONT, dp(15));
            checkView = new ImageView(MainActivity.this);
            checkView.setImageDrawable(checkIcon);
            LayoutParams cp = new LayoutParams(dp(15), dp(15));
            cp.rightMargin = dp(6);
            addView(checkView, cp);

            tv = new TextView(MainActivity.this);
            tv.setText(text);
            tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f);
            tv.setTypeface(TF_MEDIUM);
            tv.setSingleLine(true);
            addView(tv);

            setCheckedState(checked);
            setOnClickListener(v -> {
                hapticTick(v);
                onSelect.run();
            });
        }

        void setCheckedState(boolean checked) {
            checkView.setVisibility(checked ? VISIBLE : GONE);
            setPadding(checked ? dp(10) : dp(14), 0, dp(14), 0);
            if (checked) {
                tv.setTextColor(M3_ON_PRIMARY_CONT);
                setBackground(ripplePill(M3_PRIMARY_CONTAINER, 0, dp(8)));
            } else {
                tv.setTextColor(M3_ON_SURFACE);
                setBackground(ripplePill(Color.TRANSPARENT, M3_OUTLINE, dp(8)));
            }
        }
    }

    /** Liquid Material Swatch Chip showing dual-tone optical droplet preview */
    private final class SwatchChipView extends LinearLayout {
        private final TextView tv;

        SwatchChipView(String title, final int shallowCol, final int deepCol, boolean selected, Runnable onSelect) {
            super(MainActivity.this);
            setOrientation(HORIZONTAL);
            setGravity(Gravity.CENTER_VERTICAL);
            setMinimumHeight(dp(40));
            setPadding(dp(10), 0, dp(14), 0);

            View droplet = new View(MainActivity.this) {
                private final Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
                @Override
                protected void onDraw(Canvas c) {
                    float cx = getWidth() * 0.5f;
                    float cy = getHeight() * 0.5f;
                    float r = Math.min(cx, cy) - 1f;
                    p.setColor(deepCol);
                    c.drawCircle(cx, cy, r, p);
                    p.setColor(shallowCol);
                    c.drawCircle(cx - r * 0.18f, cy - r * 0.18f, r * 0.62f, p);
                    p.setColor(0xBBFFFFFF);
                    c.drawCircle(cx - r * 0.34f, cy - r * 0.34f, r * 0.20f, p);
                }
            };
            LayoutParams dpLp = new LayoutParams(dp(20), dp(20));
            dpLp.rightMargin = dp(8);
            addView(droplet, dpLp);

            tv = new TextView(MainActivity.this);
            tv.setText(title);
            tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f);
            tv.setTypeface(TF_MEDIUM);
            tv.setSingleLine(true);
            addView(tv);

            setSelectedState(selected);
            setOnClickListener(v -> {
                hapticTick(v);
                onSelect.run();
            });
        }

        void setSelectedState(boolean selected) {
            if (selected) {
                tv.setTextColor(M3_ON_PRIMARY_CONT);
                setBackground(ripplePill(M3_PRIMARY_CONTAINER, M3_PRIMARY, dp(12)));
            } else {
                tv.setTextColor(M3_ON_SURFACE);
                setBackground(ripplePill(M3_CARD_SURFACE, M3_OUTLINE, dp(12)));
            }
        }
    }

    /** Native Android Material 3 Pill Switch (52dp x 30dp animated track & thumb) */
    private final class M3Switch extends View {
        private boolean checked;
        private float progress;
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private ValueAnimator anim;
        private BoolConsumer listener;

        M3Switch(boolean initialChecked, BoolConsumer onToggle) {
            super(MainActivity.this);
            this.checked = initialChecked;
            this.progress = initialChecked ? 1f : 0f;
            this.listener = onToggle;

            setOnClickListener(v -> {
                hapticTick(v);
                setChecked(!checked, false);
            });
        }

        boolean isChecked() {
            return checked;
        }

        void setChecked(boolean target, boolean silent) {
            if (this.checked == target) return;
            this.checked = target;
            if (anim != null) anim.cancel();
            anim = ValueAnimator.ofFloat(progress, target ? 1f : 0f);
            anim.setDuration(160);
            anim.setInterpolator(new DecelerateInterpolator());
            anim.addUpdateListener(a -> {
                progress = (Float) a.getAnimatedValue();
                invalidate();
            });
            anim.start();
            if (!silent && listener != null) {
                listener.accept(checked);
            }
        }

        @Override
        protected void onMeasure(int wSpec, int hSpec) {
            setMeasuredDimension(dp(52), dp(30));
        }

        @Override
        protected void onDraw(Canvas c) {
            float w = getWidth();
            float h = getHeight();
            float r = h * 0.5f;

            int trackColor = blendColors(0xFF25303E, M3_PRIMARY, progress);
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(trackColor);
            c.drawRoundRect(dp(1), dp(1), w - dp(1), h - dp(1), r, r, paint);

            if (progress < 0.95f) {
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(dp(1.8f));
                paint.setColor(blendColors(0xFF64748B, M3_PRIMARY, progress));
                c.drawRoundRect(dp(1), dp(1), w - dp(1), h - dp(1), r, r, paint);
            }

            float minX = dp(15);
            float maxX = w - dp(15);
            float cx = minX + (maxX - minX) * progress;
            float thumbR = dp(8) + dp(3) * progress;

            paint.setStyle(Paint.Style.FILL);
            paint.setColor(blendColors(0xFF94A3B8, M3_ON_PRIMARY, progress));
            c.drawCircle(cx, h * 0.5f, thumbR, paint);
        }
    }

    private static int blendColors(int c0, int c1, float t) {
        float inv = 1f - t;
        int a = Math.round(Color.alpha(c0) * inv + Color.alpha(c1) * t);
        int r = Math.round(Color.red(c0) * inv + Color.red(c1) * t);
        int g = Math.round(Color.green(c0) * inv + Color.green(c1) * t);
        int b = Math.round(Color.blue(c0) * inv + Color.blue(c1) * t);
        return Color.argb(a, r, g, b);
    }

    // --- Layout & Card Builders ---

    private LinearLayout makePaneColumn() {
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setLayoutParams(new FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        return col;
    }

    private void addSectionHeader(LinearLayout parent, String text) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextColor(M3_ON_SURFACE_VAR);
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f);
        tv.setTypeface(TF_MEDIUM);
        tv.setPadding(dp(4), dp(2), dp(4), dp(8));
        parent.addView(tv);
    }

    private LinearLayout makeChipStrip(LinearLayout parent) {
        HorizontalScrollView hsv = new HorizontalScrollView(this);
        hsv.setHorizontalScrollBarEnabled(false);
        hsv.setClipToPadding(false);

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        hsv.addView(row);
        parent.addView(hsv);
        return row;
    }

    private LinearLayout.LayoutParams chipMargin(int index) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        if (index > 0) lp.leftMargin = dp(8);
        return lp;
    }

    private LinearLayout makeM3Card() {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(16), dp(14), dp(16), dp(14));
        card.setBackground(roundRect(M3_CARD_SURFACE, 0, dp(20)));
        return card;
    }

    private LinearLayout.LayoutParams cardTopGap() {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(12);
        return lp;
    }

    private void addCardTitle(LinearLayout card, String title) {
        TextView tv = new TextView(this);
        tv.setText(title);
        tv.setTextColor(M3_ON_SURFACE);
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f);
        tv.setTypeface(TF_MEDIUM);
        tv.setPadding(0, 0, 0, dp(10));
        card.addView(tv);
    }

    private TextView makeSubtleLabel(String text) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextColor(M3_ON_SURFACE_VAR);
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f);
        tv.setTypeface(TF_REGULAR);
        return tv;
    }

    private M3Switch addSwitchRow(LinearLayout parent, String title, String subtitle,
                                  boolean initial, BoolConsumer onToggle) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(4), 0, dp(4));

        LinearLayout texts = new LinearLayout(this);
        texts.setOrientation(LinearLayout.VERTICAL);

        TextView t = new TextView(this);
        t.setText(title);
        t.setTextColor(M3_ON_SURFACE);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14.5f);
        t.setTypeface(TF_MEDIUM);
        texts.addView(t);

        if (subtitle != null) {
            TextView sub = new TextView(this);
            sub.setText(subtitle);
            sub.setTextColor(M3_ON_SURFACE_VAR);
            sub.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f);
            sub.setPadding(0, dp(2), dp(12), 0);
            texts.addView(sub);
        }

        row.addView(texts, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));

        M3Switch sw = new M3Switch(initial, onToggle);
        row.addView(sw);

        row.setOnClickListener(v -> {
            hapticTick(v);
            sw.setChecked(!sw.isChecked(), false);
        });

        parent.addView(row);
        return sw;
    }

    private SeekBar addM3Slider(LinearLayout parent, String label, String initVal,
                                TextView badgeView, int min, int max, int initial,
                                IntConsumer onChange) {
        LinearLayout hdr = new LinearLayout(this);
        hdr.setOrientation(LinearLayout.HORIZONTAL);
        hdr.setGravity(Gravity.CENTER_VERTICAL);
        hdr.setPadding(0, dp(10), 0, dp(2));

        TextView lbl = new TextView(this);
        lbl.setText(label);
        lbl.setTextColor(M3_ON_SURFACE);
        lbl.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13.5f);
        lbl.setTypeface(TF_REGULAR);
        hdr.addView(lbl, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));

        badgeView.setText(initVal);
        badgeView.setTextColor(M3_ON_PRIMARY_CONT);
        badgeView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f);
        badgeView.setTypeface(TF_MEDIUM);
        badgeView.setPadding(dp(8), dp(2), dp(8), dp(2));
        badgeView.setBackground(roundRect(0xFF263342, 0, dp(6)));
        hdr.addView(badgeView);
        parent.addView(hdr);

        SeekBar sb = new SeekBar(this);
        sb.setMax(max - min);
        sb.setProgress(initial - min);
        sb.setProgressTintList(ColorStateList.valueOf(M3_PRIMARY));
        sb.setProgressBackgroundTintList(ColorStateList.valueOf(0xFF334155));
        sb.setThumbTintList(ColorStateList.valueOf(M3_PRIMARY));
        sb.setPadding(dp(12), dp(8), dp(12), dp(6));
        sb.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                onChange.accept(min + progress);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        parent.addView(sb);
        return sb;
    }
}
