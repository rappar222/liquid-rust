package com.liquidrust.mobile;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

/**
 * Triggered inside the newly updated app process immediately after Android replaces
 * our APK ({@link Intent#ACTION_MY_PACKAGE_REPLACED}), restarting {@link MainActivity}
 * automatically with the newly applied changes.
 */
public final class UpdateRestartReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent == null) return;
        if (Intent.ACTION_MY_PACKAGE_REPLACED.equals(intent.getAction())) {
            AppUpdater.relaunchIfPending(context);
        }
    }
}
