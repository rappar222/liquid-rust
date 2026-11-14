package com.liquidrust.mobile;

import android.app.Activity;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.content.pm.PackageInstaller;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * Zero-dependency GitHub Release checker, streamer, APK verifier, and native
 * {@link PackageInstaller} session updater for LiquidRust Android.
 */
public final class AppUpdater {

    public static final String GITHUB_REPO = "rappar222/liquid-rust";
    private static final String GITHUB_LATEST_API =
        "https://api.github.com/repos/rappar222/liquid-rust/releases/latest";

    private static final String PREFS_NAME = "liquid_updater_prefs";
    private static final String KEY_PENDING_RELAUNCH_TS = "pending_relaunch_ts";
    private static final String KEY_JUST_UPDATED_VER = "just_updated_version";
    private static final long RELAUNCH_WINDOW_MS = 5L * 60L * 1000L; // 5 minutes

    private static final Handler MAIN_HANDLER = new Handler(Looper.getMainLooper());
    private static volatile InstallErrorListener activeErrorListener;

    public static final class ReleaseInfo {
        public final String tagName;
        public final String version;
        public final String title;
        public final String notes;
        public final String assetName;
        public final String downloadUrl;
        public final long assetSize;
        public final boolean isNewer;

        public ReleaseInfo(
            String tagName,
            String version,
            String title,
            String notes,
            String assetName,
            String downloadUrl,
            long assetSize,
            boolean isNewer
        ) {
            this.tagName = tagName;
            this.version = version;
            this.title = title;
            this.notes = notes;
            this.assetName = assetName;
            this.downloadUrl = downloadUrl;
            this.assetSize = assetSize;
            this.isNewer = isNewer;
        }
    }

    public interface CheckCallback {
        void onResult(ReleaseInfo info);
        void onError(String message);
    }

    public interface DownloadCallback {
        void onProgress(long downloadedBytes, long totalBytes);
        void onInstalling();
        void onNeedsInstallPermission(File downloadedApk);
        void onError(String message);
    }

    public interface InstallErrorListener {
        void onInstallError(String message);
    }

    private File pendingPermissionApk;

    public void setInstallErrorListener(InstallErrorListener listener) {
        activeErrorListener = listener;
    }

    public boolean hasPendingPermissionApk() {
        return pendingPermissionApk != null && pendingPermissionApk.isFile();
    }

    public void clearPendingPermissionApk() {
        pendingPermissionApk = null;
    }

    public static String getCurrentVersionName(Context context) {
        try {
            PackageInfo pi = getOwnPackageInfo(context);
            if (pi != null && pi.versionName != null && !pi.versionName.isEmpty()) {
                return pi.versionName;
            }
        } catch (Throwable ignored) {}
        return "0.3.11";
    }

    @SuppressWarnings("deprecation")
    private static PackageInfo getOwnPackageInfo(Context context) throws PackageManager.NameNotFoundException {
        PackageManager pm = context.getPackageManager();
        if (Build.VERSION.SDK_INT >= 33) {
            return pm.getPackageInfo(context.getPackageName(), PackageManager.PackageInfoFlags.of(0));
        }
        return pm.getPackageInfo(context.getPackageName(), 0);
    }

    /**
     * Returns the newly installed version string once right after an update-triggered restart,
     * or {@code null} on normal launches.
     */
    public static String consumeJustUpdatedVersion(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        String ver = prefs.getString(KEY_JUST_UPDATED_VER, null);
        if (ver != null) {
            prefs.edit().remove(KEY_JUST_UPDATED_VER).apply();
        }
        cleanupCachedApks(context);
        return ver;
    }

    public static void cleanupCachedApks(Context context) {
        try {
            File dir = new File(context.getCacheDir(), "updates");
            File[] files = dir.listFiles();
            if (files != null) {
                for (File f : files) {
                    // noinspection ResultOfMethodCallIgnored
                    f.delete();
                }
            }
        } catch (Throwable ignored) {}
    }

    static void onInstallFailed(Context context, String message) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        prefs.edit().remove(KEY_PENDING_RELAUNCH_TS).apply();
        final InstallErrorListener listener = activeErrorListener;
        if (listener != null) {
            MAIN_HANDLER.post(() -> listener.onInstallError(message));
        }
    }

    static void relaunchIfPending(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        long ts = prefs.getLong(KEY_PENDING_RELAUNCH_TS, 0L);
        long now = System.currentTimeMillis();
        if (ts <= 0L || (now - ts) > RELAUNCH_WINDOW_MS) {
            return;
        }

        String newVer = getCurrentVersionName(context);
        prefs.edit()
            .remove(KEY_PENDING_RELAUNCH_TS)
            .putString(KEY_JUST_UPDATED_VER, newVer)
            .commit();

        cleanupCachedApks(context);

        Intent launch = new Intent(context, MainActivity.class);
        launch.addFlags(
            Intent.FLAG_ACTIVITY_NEW_TASK
                | Intent.FLAG_ACTIVITY_CLEAR_TOP
                | Intent.FLAG_ACTIVITY_SINGLE_TOP
        );
        try {
            context.startActivity(launch);
        } catch (Throwable ignored) {}
    }

    public void checkForUpdates(final Context context, final CheckCallback callback) {
        final String installedVer = getCurrentVersionName(context);
        Thread worker = new Thread(() -> {
            HttpURLConnection conn = null;
            try {
                URL url = new URL(GITHUB_LATEST_API);
                conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setInstanceFollowRedirects(true);
                conn.setConnectTimeout(8000);
                conn.setReadTimeout(12000);
                conn.setRequestProperty("Accept", "application/vnd.github+json");
                conn.setRequestProperty("User-Agent", "LiquidRust-Android/" + installedVer);

                int code = conn.getResponseCode();
                if (code < 200 || code >= 300) {
                    throw new Exception("GitHub returned HTTP " + code);
                }

                StringBuilder sb = new StringBuilder(8192);
                try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        sb.append(line).append('\n');
                    }
                }

                JSONObject root = new JSONObject(sb.toString());
                String tagName = root.optString("tag_name", "").trim();
                if (tagName.isEmpty()) {
                    throw new Exception("Missing release tag in GitHub response");
                }

                String releaseTitle = root.optString("name", "").trim();
                if (releaseTitle.isEmpty()) {
                    releaseTitle = "Liquid Rust " + tagName;
                }
                String body = root.optString("body", "").trim();

                JSONArray assets = root.optJSONArray("assets");
                JSONObject matchedApk = null;
                if (assets != null) {
                    for (int i = 0; i < assets.length(); i++) {
                        JSONObject a = assets.optJSONObject(i);
                        if (a == null) continue;
                        String name = a.optString("name", "");
                        if ("liquid-rust-android.apk".equalsIgnoreCase(name)) {
                            matchedApk = a;
                            break;
                        }
                        if (matchedApk == null && name.toLowerCase(Locale.US).endsWith(".apk")) {
                            matchedApk = a;
                        }
                    }
                }

                if (matchedApk == null) {
                    throw new Exception("Release " + tagName + " does not contain an Android .apk asset");
                }

                String assetName = matchedApk.optString("name", "liquid-rust-android.apk");
                String dlUrl = matchedApk.optString("browser_download_url", "");
                long size = matchedApk.optLong("size", 0L);
                if (dlUrl.isEmpty()) {
                    throw new Exception("Missing APK download URL");
                }

                String remoteVer = normalizeVersion(tagName);
                boolean newer = compareVersions(remoteVer, installedVer) > 0;

                final ReleaseInfo info = new ReleaseInfo(
                    tagName,
                    remoteVer,
                    releaseTitle,
                    body,
                    assetName,
                    dlUrl,
                    size,
                    newer
                );
                MAIN_HANDLER.post(() -> callback.onResult(info));
            } catch (final Throwable t) {
                final String msg = (t.getMessage() != null && !t.getMessage().isEmpty())
                    ? t.getMessage()
                    : "Network error checking GitHub Releases";
                MAIN_HANDLER.post(() -> callback.onError(msg));
            } finally {
                if (conn != null) {
                    conn.disconnect();
                }
            }
        }, "LiquidRust-UpdateCheck");
        worker.setDaemon(true);
        worker.start();
    }

    public void downloadAndInstall(
        final Activity activity,
        final ReleaseInfo info,
        final DownloadCallback callback
    ) {
        final Context appCtx = activity.getApplicationContext();
        final String installedVer = getCurrentVersionName(appCtx);

        Thread worker = new Thread(() -> {
            HttpURLConnection conn = null;
            try {
                File dir = new File(appCtx.getCacheDir(), "updates");
                if (!dir.exists() && !dir.mkdirs()) {
                    throw new Exception("Could not create update cache folder");
                }
                File apkFile = new File(dir, "liquid-rust-update.apk");
                if (apkFile.exists()) {
                    // noinspection ResultOfMethodCallIgnored
                    apkFile.delete();
                }

                conn = openFollowingRedirects(info.downloadUrl, installedVer);
                long contentLen = conn.getContentLengthLong();
                final long total = contentLen > 0 ? contentLen : info.assetSize;

                byte[] buf = new byte[16 * 1024];
                long downloaded = 0L;
                long lastUiTick = 0L;

                try (InputStream in = conn.getInputStream();
                     FileOutputStream out = new FileOutputStream(apkFile)) {
                    int n;
                    while ((n = in.read(buf)) != -1) {
                        out.write(buf, 0, n);
                        downloaded += n;
                        long now = System.currentTimeMillis();
                        if (now - lastUiTick >= 32L || downloaded == total) {
                            lastUiTick = now;
                            final long dlSnap = downloaded;
                            final long totSnap = Math.max(total, downloaded);
                            MAIN_HANDLER.post(() -> callback.onProgress(dlSnap, totSnap));
                        }
                    }
                    out.getFD().sync();
                }

                verifyDownloadedApk(appCtx, apkFile);

                MAIN_HANDLER.post(() -> {
                    if (!activity.getPackageManager().canRequestPackageInstalls()) {
                        pendingPermissionApk = apkFile;
                        callback.onNeedsInstallPermission(apkFile);
                    } else {
                        pendingPermissionApk = null;
                        callback.onInstalling();
                        commitPackageInstallerSession(activity, apkFile, callback);
                    }
                });
            } catch (final Throwable t) {
                final String msg = (t.getMessage() != null && !t.getMessage().isEmpty())
                    ? t.getMessage()
                    : "Failed downloading update APK";
                MAIN_HANDLER.post(() -> callback.onError(msg));
            } finally {
                if (conn != null) {
                    conn.disconnect();
                }
            }
        }, "LiquidRust-UpdateDownload");
        worker.setDaemon(true);
        worker.start();
    }

    public void continuePendingInstallIfPermitted(Activity activity, DownloadCallback callback) {
        if (pendingPermissionApk == null || !pendingPermissionApk.isFile()) {
            return;
        }
        if (!activity.getPackageManager().canRequestPackageInstalls()) {
            return;
        }
        File apk = pendingPermissionApk;
        pendingPermissionApk = null;
        callback.onInstalling();
        commitPackageInstallerSession(activity, apk, callback);
    }

    public static void openUnknownSourcesSettings(Activity activity) {
        Intent intent = new Intent(
            Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
            Uri.parse("package:" + activity.getPackageName())
        );
        try {
            activity.startActivity(intent);
        } catch (Throwable ignored) {
            activity.startActivity(new Intent(Settings.ACTION_SECURITY_SETTINGS));
        }
    }

    private static void commitPackageInstallerSession(
        final Activity activity,
        final File apkFile,
        final DownloadCallback callback
    ) {
        final Context appCtx = activity.getApplicationContext();
        Thread installerThread = new Thread(() -> {
            PackageInstaller.Session session = null;
            try {
                // Record pending auto-relaunch before committing so UpdateRestartReceiver knows to reopen MainActivity
                SharedPreferences prefs = appCtx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
                prefs.edit()
                    .putLong(KEY_PENDING_RELAUNCH_TS, System.currentTimeMillis())
                    .commit();

                PackageInstaller installer = appCtx.getPackageManager().getPackageInstaller();
                PackageInstaller.SessionParams params =
                    new PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL);
                params.setAppPackageName(appCtx.getPackageName());
                if (Build.VERSION.SDK_INT >= 31) {
                    params.setRequireUserAction(PackageInstaller.SessionParams.USER_ACTION_NOT_REQUIRED);
                }

                int sessionId = installer.createSession(params);
                session = installer.openSession(sessionId);

                try (InputStream in = new FileInputStream(apkFile);
                     OutputStream out = session.openWrite("liquid-rust-update.apk", 0, apkFile.length())) {
                    byte[] buf = new byte[32 * 1024];
                    int n;
                    while ((n = in.read(buf)) != -1) {
                        out.write(buf, 0, n);
                    }
                    session.fsync(out);
                }

                Intent statusIntent = new Intent(appCtx, UpdateInstallReceiver.class);
                statusIntent.setPackage(appCtx.getPackageName());

                int piFlags = PendingIntent.FLAG_UPDATE_CURRENT;
                if (Build.VERSION.SDK_INT >= 31) {
                    piFlags |= PendingIntent.FLAG_MUTABLE;
                }

                PendingIntent pi = PendingIntent.getBroadcast(
                    appCtx,
                    sessionId,
                    statusIntent,
                    piFlags
                );
                session.commit(pi.getIntentSender());
                session.close();
                session = null;
            } catch (final Throwable t) {
                if (session != null) {
                    try { session.abandon(); } catch (Throwable ignored) {}
                }
                SharedPreferences prefs = appCtx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
                prefs.edit().remove(KEY_PENDING_RELAUNCH_TS).apply();

                final String msg = "Installer error: " + t.getMessage();
                MAIN_HANDLER.post(() -> callback.onError(msg));
            }
        }, "LiquidRust-PackageCommit");
        installerThread.setDaemon(true);
        installerThread.start();
    }

    private static HttpURLConnection openFollowingRedirects(String urlStr, String ver) throws Exception {
        String current = urlStr;
        for (int hop = 0; hop < 6; hop++) {
            HttpURLConnection c = (HttpURLConnection) new URL(current).openConnection();
            c.setInstanceFollowRedirects(false);
            c.setConnectTimeout(10000);
            c.setReadTimeout(30000);
            c.setRequestProperty("User-Agent", "LiquidRust-Android/" + ver);
            int status = c.getResponseCode();
            if (status == HttpURLConnection.HTTP_MOVED_TEMP
                || status == HttpURLConnection.HTTP_MOVED_PERM
                || status == HttpURLConnection.HTTP_SEE_OTHER
                || status == 307
                || status == 308) {
                String next = c.getHeaderField("Location");
                c.disconnect();
                if (next == null || next.isEmpty()) {
                    throw new Exception("Redirect missing Location header");
                }
                current = new URL(new URL(current), next).toString();
                continue;
            }
            if (status < 200 || status >= 300) {
                c.disconnect();
                throw new Exception("Asset download failed with HTTP " + status);
            }
            return c;
        }
        throw new Exception("Too many HTTP redirects downloading APK");
    }

    @SuppressWarnings("deprecation")
    private static void verifyDownloadedApk(Context ctx, File apkFile) throws Exception {
        if (!apkFile.isFile() || apkFile.length() < 32 * 1024) {
            throw new Exception("Downloaded APK is incomplete (" + apkFile.length() + " bytes)");
        }
        byte[] magic = new byte[4];
        try (FileInputStream fis = new FileInputStream(apkFile)) {
            if (fis.read(magic) != 4
                || magic[0] != 'P' || magic[1] != 'K' || magic[2] != 3 || magic[3] != 4) {
                throw new Exception("Downloaded file is not a valid APK archive");
            }
        }
        PackageManager pm = ctx.getPackageManager();
        PackageInfo archiveInfo = (Build.VERSION.SDK_INT >= 33)
            ? pm.getPackageArchiveInfo(apkFile.getAbsolutePath(), PackageManager.PackageInfoFlags.of(0))
            : pm.getPackageArchiveInfo(apkFile.getAbsolutePath(), 0);
        if (archiveInfo == null || !ctx.getPackageName().equals(archiveInfo.packageName)) {
            throw new Exception("APK package ID mismatch");
        }
    }

    public static String normalizeVersion(String tag) {
        if (tag == null) return "0.0.0";
        String s = tag.trim();
        while (!s.isEmpty() && (s.charAt(0) == 'v' || s.charAt(0) == 'V')) {
            s = s.substring(1);
        }
        return s.trim();
    }

    public static int compareVersions(String a, String b) {
        int[] va = parseSemVer(a);
        int[] vb = parseSemVer(b);
        for (int i = 0; i < 3; i++) {
            if (va[i] != vb[i]) {
                return Integer.compare(va[i], vb[i]);
            }
        }
        return 0;
    }

    private static int[] parseSemVer(String v) {
        String clean = normalizeVersion(v);
        int dash = clean.indexOf('-');
        if (dash >= 0) clean = clean.substring(0, dash);
        int plus = clean.indexOf('+');
        if (plus >= 0) clean = clean.substring(0, plus);

        String[] parts = clean.split("\\.");
        int[] out = new int[]{ 0, 0, 0 };
        for (int i = 0; i < Math.min(3, parts.length); i++) {
            try {
                out[i] = Integer.parseInt(parts[i].trim());
            } catch (NumberFormatException ignored) {
                out[i] = 0;
            }
        }
        return out;
    }

    public static String formatBytes(long bytes) {
        if (bytes >= 1024L * 1024L) {
            return String.format(Locale.US, "%.2f MB", bytes / (1024.0 * 1024.0));
        } else if (bytes >= 1024L) {
            return String.format(Locale.US, "%.0f KB", bytes / 1024.0);
        }
        return bytes + " B";
    }
}
