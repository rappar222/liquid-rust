package com.liquidrust.mobile;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInstaller;
import android.os.Build;

/**
 * Receives asynchronous status callbacks from Android's {@link PackageInstaller.Session}.
 * Launches the system confirmation prompt when required and reports failures back to the UI.
 */
public final class UpdateInstallReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent == null) return;

        int status = intent.getIntExtra(
            PackageInstaller.EXTRA_STATUS,
            PackageInstaller.STATUS_FAILURE
        );

        if (status == PackageInstaller.STATUS_PENDING_USER_ACTION) {
            Intent confirmIntent = extractExtraIntent(intent);
            if (confirmIntent != null) {
                confirmIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                try {
                    context.startActivity(confirmIntent);
                } catch (Throwable t) {
                    AppUpdater.onInstallFailed(context, "Could not open installer dialog: " + t.getMessage());
                }
            } else {
                AppUpdater.onInstallFailed(context, "Installer confirmation intent missing.");
            }
        } else if (status == PackageInstaller.STATUS_SUCCESS) {
            AppUpdater.relaunchIfPending(context);
        } else if (status >= PackageInstaller.STATUS_FAILURE) {
            String msg = intent.getStringExtra(PackageInstaller.EXTRA_STATUS_MESSAGE);
            if (status == PackageInstaller.STATUS_FAILURE_ABORTED) {
                AppUpdater.onInstallFailed(context, "Update cancelled.");
            } else if (status == PackageInstaller.STATUS_FAILURE_CONFLICT) {
                AppUpdater.onInstallFailed(context,
                    "Signing certificate conflict with existing install. Uninstall old build once, then install the new APK.");
            } else {
                AppUpdater.onInstallFailed(context,
                    (msg != null && !msg.isEmpty()) ? ("Install failed: " + msg) : "Update installation failed.");
            }
        }
    }

    @SuppressWarnings("deprecation")
    private static Intent extractExtraIntent(Intent intent) {
        if (Build.VERSION.SDK_INT >= 33) {
            return intent.getParcelableExtra(Intent.EXTRA_INTENT, Intent.class);
        }
        return intent.getParcelableExtra(Intent.EXTRA_INTENT);
    }
}
