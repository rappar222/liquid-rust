package com.liquidrust.mobile;

import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.drawable.Drawable;

/**
 * Crisp 24dp Material Design 3 monochrome vector icon Drawable.
 * Eliminates colored emoji bitmaps for an authentic Android Material You UI.
 */
public final class MaterialIconDrawable extends Drawable {

    public static final int ICON_PLAY = 0;
    public static final int ICON_PAUSE = 1;
    public static final int ICON_STEP = 2;
    public static final int ICON_RESET = 3;
    public static final int ICON_TILT = 4;
    public static final int ICON_TUNE = 5;
    public static final int ICON_PULL = 6;
    public static final int ICON_PUSH = 7;
    public static final int ICON_ORBIT = 8;
    public static final int ICON_CENTER = 9;
    public static final int ICON_MODE_2D = 10;
    public static final int ICON_MODE_3D = 11;
    public static final int ICON_CHECK = 12;
    public static final int ICON_CHEVRON_DOWN = 13;

    private int iconType;
    private int tintColor;
    private final int sizePx;
    private final Paint strokePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path path = new Path();
    private final RectF oval = new RectF();

    public MaterialIconDrawable(int iconType, int color, int sizePx) {
        this.iconType = iconType;
        this.tintColor = color;
        this.sizePx = sizePx;

        strokePaint.setStyle(Paint.Style.STROKE);
        strokePaint.setStrokeCap(Paint.Cap.ROUND);
        strokePaint.setStrokeJoin(Paint.Join.ROUND);
        strokePaint.setColor(color);

        fillPaint.setStyle(Paint.Style.FILL);
        fillPaint.setColor(color);
    }

    public void setIconType(int iconType) {
        if (this.iconType != iconType) {
            this.iconType = iconType;
            invalidateSelf();
        }
    }

    public void setTintColor(int color) {
        if (this.tintColor != color) {
            this.tintColor = color;
            strokePaint.setColor(color);
            fillPaint.setColor(color);
            invalidateSelf();
        }
    }

    @Override
    public int getIntrinsicWidth() {
        return sizePx;
    }

    @Override
    public int getIntrinsicHeight() {
        return sizePx;
    }

    @Override
    public void draw(Canvas canvas) {
        Rect b = getBounds();
        if (b.isEmpty()) return;

        float w = b.width();
        float h = b.height();
        float scale = Math.min(w, h) / 24.0f;

        int save = canvas.save();
        canvas.translate(b.left + (w - 24f * scale) * 0.5f, b.top + (h - 24f * scale) * 0.5f);
        canvas.scale(scale, scale);

        strokePaint.setStrokeWidth(2.15f);
        path.reset();

        switch (iconType) {
            case ICON_PLAY: {
                path.moveTo(8.0f, 5.4f);
                path.lineTo(19.0f, 12.0f);
                path.lineTo(8.0f, 18.6f);
                path.close();
                canvas.drawPath(path, fillPaint);
                strokePaint.setStrokeWidth(1.6f);
                canvas.drawPath(path, strokePaint);
                break;
            }
            case ICON_PAUSE: {
                oval.set(6.5f, 5.2f, 10.3f, 18.8f);
                canvas.drawRoundRect(oval, 1.5f, 1.5f, fillPaint);
                oval.set(13.7f, 5.2f, 17.5f, 18.8f);
                canvas.drawRoundRect(oval, 1.5f, 1.5f, fillPaint);
                break;
            }
            case ICON_STEP: {
                path.moveTo(6.2f, 6.2f);
                path.lineTo(14.8f, 12.0f);
                path.lineTo(6.2f, 17.8f);
                path.close();
                canvas.drawPath(path, fillPaint);
                oval.set(16.2f, 6.0f, 18.4f, 18.0f);
                canvas.drawRoundRect(oval, 1.1f, 1.1f, fillPaint);
                break;
            }
            case ICON_RESET: {
                oval.set(4.8f, 4.8f, 19.2f, 19.2f);
                canvas.drawArc(oval, -50f, 290f, false, strokePaint);
                path.moveTo(14.5f, 3.8f);
                path.lineTo(19.2f, 6.2f);
                path.lineTo(15.6f, 9.8f);
                canvas.drawPath(path, strokePaint);
                break;
            }
            case ICON_TILT: {
                // Tilted smartphone outline + motion arc
                int s2 = canvas.save();
                canvas.rotate(-22f, 12f, 12f);
                oval.set(7.5f, 4.5f, 16.5f, 19.5f);
                canvas.drawRoundRect(oval, 2.2f, 2.2f, strokePaint);
                canvas.drawCircle(12f, 16.8f, 0.9f, fillPaint);
                canvas.restoreToCount(s2);

                strokePaint.setStrokeWidth(1.85f);
                oval.set(2.5f, 2.5f, 21.5f, 21.5f);
                canvas.drawArc(oval, -150f, 52f, false, strokePaint);
                canvas.drawArc(oval, 30f, 52f, false, strokePaint);
                break;
            }
            case ICON_TUNE: {
                // Material 'tune' 3-slider icon
                canvas.drawLine(4f, 7f, 20f, 7f, strokePaint);
                canvas.drawLine(4f, 12f, 20f, 12f, strokePaint);
                canvas.drawLine(4f, 17f, 20f, 17f, strokePaint);
                canvas.drawCircle(9f, 7f, 2.3f, fillPaint);
                canvas.drawCircle(15.5f, 12f, 2.3f, fillPaint);
                canvas.drawCircle(10.5f, 17f, 2.3f, fillPaint);
                break;
            }
            case ICON_PULL: {
                // Concentric ripple + center pull dot
                canvas.drawCircle(12f, 12f, 2.6f, fillPaint);
                strokePaint.setStrokeWidth(1.9f);
                oval.set(5.2f, 5.2f, 18.8f, 18.8f);
                canvas.drawArc(oval, 15f, 60f, false, strokePaint);
                canvas.drawArc(oval, 105f, 60f, false, strokePaint);
                canvas.drawArc(oval, 195f, 60f, false, strokePaint);
                canvas.drawArc(oval, 285f, 60f, false, strokePaint);
                break;
            }
            case ICON_PUSH: {
                // Outward burst ring + rays
                strokePaint.setStrokeWidth(2.0f);
                canvas.drawCircle(12f, 12f, 3.6f, strokePaint);
                canvas.drawLine(12f, 3.5f, 12f, 6.2f, strokePaint);
                canvas.drawLine(12f, 17.8f, 12f, 20.5f, strokePaint);
                canvas.drawLine(3.5f, 12f, 6.2f, 12f, strokePaint);
                canvas.drawLine(17.8f, 12f, 20.5f, 12f, strokePaint);
                canvas.drawLine(6.0f, 6.0f, 7.9f, 7.9f, strokePaint);
                canvas.drawLine(16.1f, 16.1f, 18.0f, 18.0f, strokePaint);
                canvas.drawLine(18.0f, 6.0f, 16.1f, 7.9f, strokePaint);
                canvas.drawLine(7.9f, 16.1f, 6.0f, 18.0f, strokePaint);
                break;
            }
            case ICON_ORBIT: {
                // 3D globe + orbital equator/meridian
                strokePaint.setStrokeWidth(1.85f);
                canvas.drawCircle(12f, 12f, 7.6f, strokePaint);
                oval.set(4.4f, 8.8f, 19.6f, 15.2f);
                canvas.drawOval(oval, strokePaint);
                oval.set(8.8f, 4.4f, 15.2f, 19.6f);
                canvas.drawOval(oval, strokePaint);
                break;
            }
            case ICON_CENTER: {
                // Center focus brackets + dot
                canvas.drawCircle(12f, 12f, 2.4f, fillPaint);
                path.moveTo(4.5f, 8.8f); path.lineTo(4.5f, 4.5f); path.lineTo(8.8f, 4.5f);
                path.moveTo(15.2f, 4.5f); path.lineTo(19.5f, 4.5f); path.lineTo(19.5f, 8.8f);
                path.moveTo(19.5f, 15.2f); path.lineTo(19.5f, 19.5f); path.lineTo(15.2f, 19.5f);
                path.moveTo(8.8f, 19.5f); path.lineTo(4.5f, 19.5f); path.lineTo(4.5f, 15.2f);
                canvas.drawPath(path, strokePaint);
                break;
            }
            case ICON_MODE_2D: {
                oval.set(4.8f, 4.8f, 19.2f, 19.2f);
                canvas.drawRoundRect(oval, 3.0f, 3.0f, strokePaint);
                strokePaint.setStrokeWidth(1.7f);
                path.moveTo(5.2f, 14.2f);
                path.quadTo(9.0f, 11.4f, 12.0f, 13.8f);
                path.quadTo(15.0f, 16.2f, 18.8f, 13.2f);
                canvas.drawPath(path, strokePaint);
                break;
            }
            case ICON_MODE_3D: {
                // Isometric 3D cube
                strokePaint.setStrokeWidth(1.9f);
                path.moveTo(12f, 3.8f);
                path.lineTo(19.2f, 7.9f);
                path.lineTo(19.2f, 16.1f);
                path.lineTo(12f, 20.2f);
                path.lineTo(4.8f, 16.1f);
                path.lineTo(4.8f, 7.9f);
                path.close();
                canvas.drawPath(path, strokePaint);
                path.reset();
                path.moveTo(4.8f, 7.9f);
                path.lineTo(12f, 12f);
                path.lineTo(19.2f, 7.9f);
                path.moveTo(12f, 12f);
                path.lineTo(12f, 20.2f);
                canvas.drawPath(path, strokePaint);
                break;
            }
            case ICON_CHECK: {
                strokePaint.setStrokeWidth(2.4f);
                path.moveTo(5.2f, 12.5f);
                path.lineTo(9.8f, 17.0f);
                path.lineTo(18.8f, 7.4f);
                canvas.drawPath(path, strokePaint);
                break;
            }
            case ICON_CHEVRON_DOWN:
            default: {
                strokePaint.setStrokeWidth(2.3f);
                path.moveTo(6.2f, 9.5f);
                path.lineTo(12.0f, 15.2f);
                path.lineTo(17.8f, 9.5f);
                canvas.drawPath(path, strokePaint);
                break;
            }
        }

        canvas.restoreToCount(save);
    }

    @Override
    public void setAlpha(int alpha) {
        strokePaint.setAlpha(alpha);
        fillPaint.setAlpha(alpha);
    }

    @Override
    public void setColorFilter(ColorFilter colorFilter) {
        strokePaint.setColorFilter(colorFilter);
        fillPaint.setColorFilter(colorFilter);
    }

    @Override
    @SuppressWarnings("deprecation")
    public int getOpacity() {
        return PixelFormat.TRANSLUCENT;
    }
}
