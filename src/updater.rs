use eframe::egui;
use serde::Deserialize;
use std::cmp::Ordering;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GITHUB_REPO: &str = "rappar222/liquid-rust";
pub const GITHUB_LATEST_RELEASE_API: &str =
    "https://api.github.com/repos/rappar222/liquid-rust/releases/latest";

#[derive(Deserialize, Debug)]
struct GithubAssetJson {
    name: String,
    browser_download_url: String,
    size: u64,
}

#[derive(Deserialize, Debug)]
struct GithubReleaseJson {
    tag_name: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    body: String,
    #[serde(default)]
    html_url: String,
    #[serde(default)]
    assets: Vec<GithubAssetJson>,
}

#[derive(Clone, Debug)]
pub struct ReleaseInfo {
    pub tag_name: String,
    pub version: String,
    pub title: String,
    pub notes: String,
    pub html_url: String,
    pub asset_name: String,
    pub download_url: String,
    pub asset_size: u64,
    pub is_newer: bool,
}

#[derive(Clone, Debug)]
pub enum UpdatePhase {
    Idle,
    Checking,
    UpToDate(ReleaseInfo),
    UpdateAvailable(ReleaseInfo),
    Downloading {
        info: ReleaseInfo,
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    Installing(ReleaseInfo),
    ReadyToRestart {
        version: String,
        target_exe: PathBuf,
    },
    Error(String),
}

#[derive(Clone, Debug)]
pub struct UpdaterSnapshot {
    pub phase: UpdatePhase,
    pub show_modal: bool,
    pub banner_dismissed: bool,
}

#[derive(Clone)]
pub struct UpdaterHandle {
    inner: Arc<Mutex<UpdaterSnapshot>>,
}

impl Default for UpdaterHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl UpdaterHandle {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(UpdaterSnapshot {
                phase: UpdatePhase::Idle,
                show_modal: false,
                banner_dismissed: false,
            })),
        }
    }

    pub fn snapshot(&self) -> UpdaterSnapshot {
        self.inner
            .lock()
            .map(|g| g.clone())
            .unwrap_or(UpdaterSnapshot {
                phase: UpdatePhase::Idle,
                show_modal: false,
                banner_dismissed: false,
            })
    }

    pub fn open_modal(&self) {
        if let Ok(mut g) = self.inner.lock() {
            g.show_modal = true;
        }
    }

    pub fn close_modal(&self) {
        if let Ok(mut g) = self.inner.lock() {
            g.show_modal = false;
            if matches!(g.phase, UpdatePhase::UpdateAvailable(_)) {
                g.banner_dismissed = true;
            }
        }
    }

    /// Spawn a non-blocking background check against GitHub Releases.
    /// If `manual` is true, opens the Update modal immediately to show status.
    pub fn spawn_check(&self, ctx: Option<egui::Context>, manual: bool) {
        {
            let Ok(mut g) = self.inner.lock() else {
                return;
            };
            if matches!(
                g.phase,
                UpdatePhase::Checking
                    | UpdatePhase::Downloading { .. }
                    | UpdatePhase::Installing(_)
            ) {
                if manual {
                    g.show_modal = true;
                }
                return;
            }
            g.phase = UpdatePhase::Checking;
            if manual {
                g.show_modal = true;
                g.banner_dismissed = false;
            }
        }

        let state = Arc::clone(&self.inner);
        std::thread::spawn(move || {
            let result = fetch_latest_release();
            if let Ok(mut g) = state.lock() {
                match result {
                    Ok(info) => {
                        if info.is_newer {
                            g.phase = UpdatePhase::UpdateAvailable(info);
                            if !g.banner_dismissed || manual {
                                g.show_modal = true;
                            }
                        } else {
                            g.phase = UpdatePhase::UpToDate(info);
                            if manual {
                                g.show_modal = true;
                            }
                        }
                    }
                    Err(err) => {
                        if manual {
                            g.phase = UpdatePhase::Error(err);
                            g.show_modal = true;
                        } else {
                            // Fail silently on automatic startup checks (e.g. offline laptop)
                            g.phase = UpdatePhase::Idle;
                        }
                    }
                }
            }
            if let Some(c) = ctx {
                c.request_repaint();
            }
        });
    }

    /// Download the selected release asset in the background, atomically install it,
    /// and transition to `ReadyToRestart` so the GUI immediately relaunches into the new binary.
    pub fn spawn_download_and_apply(&self, info: ReleaseInfo, ctx: Option<egui::Context>) {
        {
            let Ok(mut g) = self.inner.lock() else {
                return;
            };
            if matches!(
                g.phase,
                UpdatePhase::Downloading { .. } | UpdatePhase::Installing(_)
            ) {
                return;
            }
            g.show_modal = true;
            g.phase = UpdatePhase::Downloading {
                total_bytes: info.asset_size,
                downloaded_bytes: 0,
                info: info.clone(),
            };
        }

        let state = Arc::clone(&self.inner);
        std::thread::spawn(move || {
            let progress_state = Arc::clone(&state);
            let progress_ctx = ctx.clone();
            let progress_info = info.clone();

            let download_res = download_asset_bytes(&info.download_url, info.asset_size, |dl, tot| {
                if let Ok(mut g) = progress_state.lock() {
                    g.phase = UpdatePhase::Downloading {
                        info: progress_info.clone(),
                        downloaded_bytes: dl,
                        total_bytes: tot,
                    };
                }
                if let Some(ref c) = progress_ctx {
                    c.request_repaint();
                }
            });

            let raw_bytes = match download_res {
                Ok(b) => b,
                Err(e) => {
                    if let Ok(mut g) = state.lock() {
                        g.phase = UpdatePhase::Error(format!("Download failed: {}", e));
                    }
                    if let Some(c) = ctx {
                        c.request_repaint();
                    }
                    return;
                }
            };

            if let Ok(mut g) = state.lock() {
                g.phase = UpdatePhase::Installing(info.clone());
            }
            if let Some(ref c) = ctx {
                c.request_repaint();
            }

            let install_res = extract_and_install_binary(&info.asset_name, &raw_bytes);
            if let Ok(mut g) = state.lock() {
                match install_res {
                    Ok(target_exe) => {
                        g.phase = UpdatePhase::ReadyToRestart {
                            version: info.version.clone(),
                            target_exe,
                        };
                    }
                    Err(e) => {
                        g.phase = UpdatePhase::Error(format!("Install failed: {}", e));
                    }
                }
            }
            if let Some(c) = ctx {
                c.request_repaint();
            }
        });
    }
}

/// Normalize version strings like `"v0.3.2"` or `"0.3.2-beta.1"` into `(major, minor, patch)`.
pub fn parse_semver(ver: &str) -> Option<(u32, u32, u32)> {
    let trimmed = ver.trim().trim_start_matches(['v', 'V']);
    let core = trimmed.split(['-', '+']).next()?.trim();
    let mut parts = core.split('.');
    let major = parts.next()?.parse::<u32>().ok()?;
    let minor = parts.next().unwrap_or("0").parse::<u32>().ok()?;
    let patch = parts.next().unwrap_or("0").parse::<u32>().ok()?;
    Some((major, minor, patch))
}

pub fn normalize_version_str(tag: &str) -> String {
    tag.trim().trim_start_matches(['v', 'V']).to_string()
}

pub fn compare_versions(a: &str, b: &str) -> Ordering {
    match (parse_semver(a), parse_semver(b)) {
        (Some(va), Some(vb)) => va.cmp(&vb),
        _ => normalize_version_str(a).cmp(&normalize_version_str(b)),
    }
}

/// Remove any leftover `.old` or `.update.tmp` files created by previous self-updates.
pub fn cleanup_old_binary() {
    if let Ok(exe) = resolve_target_executable_path() {
        let old_path = exe.with_extension(if cfg!(windows) { "exe.old" } else { "old" });
        let _ = fs::remove_file(&old_path);
        if let Some(parent) = exe.parent() {
            let _ = fs::remove_file(parent.join(".liquid-rust.update.tmp"));
        }
    }
}

/// Resolve the actual on-disk binary to replace and relaunch (handling AppImage & Linux `/proc/self/exe` quirks).
pub fn resolve_target_executable_path() -> Result<PathBuf, String> {
    #[cfg(target_os = "linux")]
    {
        if let Some(appimage) = std::env::var_os("APPIMAGE") {
            let p = PathBuf::from(appimage);
            if p.exists() {
                return Ok(p);
            }
        }
    }

    let exe = std::env::current_exe()
        .map_err(|e| format!("Failed to locate current executable: {}", e))?;

    #[cfg(target_os = "linux")]
    {
        let s = exe.to_string_lossy();
        if let Some(clean) = s.strip_suffix(" (deleted)") {
            return Ok(PathBuf::from(clean));
        }
    }

    Ok(exe)
}

fn candidate_asset_names() -> Vec<&'static str> {
    #[cfg(target_os = "linux")]
    {
        let is_appimage = std::env::var_os("APPIMAGE").is_some();
        #[cfg(target_arch = "x86_64")]
        {
            if is_appimage {
                return vec![
                    "liquid-rust-x86_64.AppImage",
                    "liquid-rust-linux-x86_64.tar.gz",
                    "liquid-rust",
                ];
            }
            return vec!["liquid-rust-linux-x86_64.tar.gz", "liquid-rust"];
        }
        #[cfg(target_arch = "aarch64")]
        {
            let _ = is_appimage;
            return vec!["liquid-rust-linux-aarch64.tar.gz", "liquid-rust-aarch64"];
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = is_appimage;
            return vec!["liquid-rust"];
        }
    }

    #[cfg(target_os = "windows")]
    {
        return vec!["liquid-rust-windows-x86_64.zip", "liquid-rust.exe"];
    }

    #[cfg(target_os = "macos")]
    {
        #[cfg(target_arch = "aarch64")]
        {
            return vec![
                "liquid-rust-macos-arm64.tar.gz",
                "liquid-rust-macos-x86_64.tar.gz",
            ];
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            return vec![
                "liquid-rust-macos-x86_64.tar.gz",
                "liquid-rust-macos-arm64.tar.gz",
            ];
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        vec!["liquid-rust"]
    }
}

fn build_http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(8))
        .timeout_read(Duration::from_secs(30))
        .user_agent(&format!("LiquidRust-Desktop/{}", CURRENT_VERSION))
        .build()
}

pub fn fetch_latest_release() -> Result<ReleaseInfo, String> {
    let agent = build_http_agent();
    let resp = agent
        .get(GITHUB_LATEST_RELEASE_API)
        .set("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| format!("GitHub API request failed: {}", e))?;

    let body_str = resp
        .into_string()
        .map_err(|e| format!("Failed reading GitHub API response: {}", e))?;

    let parsed: GithubReleaseJson = serde_json::from_str(&body_str)
        .map_err(|e| format!("Invalid GitHub release JSON: {}", e))?;

    let candidates = candidate_asset_names();
    let mut matched_asset: Option<&GithubAssetJson> = None;
    for want in &candidates {
        if let Some(found) = parsed
            .assets
            .iter()
            .find(|a| a.name.eq_ignore_ascii_case(want))
        {
            matched_asset = Some(found);
            break;
        }
    }

    let asset = matched_asset.ok_or_else(|| {
        format!(
            "No compatible binary asset found in release {} (expected one of: {})",
            parsed.tag_name,
            candidates.join(", ")
        )
    })?;

    let remote_ver = normalize_version_str(&parsed.tag_name);
    let is_newer = compare_versions(&remote_ver, CURRENT_VERSION) == Ordering::Greater;
    let title = if parsed.name.trim().is_empty() {
        format!("LiquidRust {}", parsed.tag_name)
    } else {
        parsed.name.trim().to_string()
    };

    Ok(ReleaseInfo {
        tag_name: parsed.tag_name,
        version: remote_ver,
        title,
        notes: parsed.body.trim().to_string(),
        html_url: parsed.html_url,
        asset_name: asset.name.clone(),
        download_url: asset.browser_download_url.clone(),
        asset_size: asset.size,
        is_newer,
    })
}

fn download_asset_bytes<F>(url: &str, expected_size: u64, mut on_progress: F) -> Result<Vec<u8>, String>
where
    F: FnMut(u64, u64),
{
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(120))
        .user_agent(&format!("LiquidRust-Desktop/{}", CURRENT_VERSION))
        .build();

    let resp = agent
        .get(url)
        .call()
        .map_err(|e| format!("HTTP error fetching asset: {}", e))?;

    let total = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(expected_size);

    let mut reader = resp.into_reader();
    let mut out = Vec::with_capacity(total.min(64 * 1024 * 1024) as usize);
    let mut buf = [0u8; 32 * 1024];
    let mut downloaded: u64 = 0;

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("Network stream interrupted: {}", e))?;
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
        downloaded += n as u64;
        on_progress(downloaded, total.max(downloaded));
    }

    if out.len() < 64 * 1024 {
        return Err(format!(
            "Downloaded payload is unexpectedly small ({} bytes)",
            out.len()
        ));
    }

    Ok(out)
}

fn extract_binary_from_archive(asset_name: &str, raw_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let lower = asset_name.to_ascii_lowercase();
    let expected_bin = if cfg!(windows) {
        "liquid-rust.exe"
    } else {
        "liquid-rust"
    };

    if lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
        let gz = flate2::read::GzDecoder::new(Cursor::new(raw_bytes));
        let mut archive = tar::Archive::new(gz);
        let entries = archive
            .entries()
            .map_err(|e| format!("Corrupt tar.gz archive: {}", e))?;

        for entry_res in entries {
            let mut entry = entry_res.map_err(|e| format!("Tar entry error: {}", e))?;
            let path = entry
                .path()
                .map_err(|e| format!("Invalid tar path: {}", e))?
                .into_owned();
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if file_name.eq_ignore_ascii_case(expected_bin)
                || file_name.eq_ignore_ascii_case("liquid-rust")
            {
                let mut bin = Vec::new();
                entry
                    .read_to_end(&mut bin)
                    .map_err(|e| format!("Failed extracting binary from tar.gz: {}", e))?;
                return Ok(bin);
            }
        }
        return Err(format!(
            "Archive '{}' did not contain '{}'",
            asset_name, expected_bin
        ));
    }

    if lower.ends_with(".zip") {
        let mut zip = zip::ZipArchive::new(Cursor::new(raw_bytes))
            .map_err(|e| format!("Corrupt zip archive: {}", e))?;
        for i in 0..zip.len() {
            let mut file = zip
                .by_index(i)
                .map_err(|e| format!("Zip entry error: {}", e))?;
            let enclosed = file.mangled_name();
            let file_name = enclosed
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if file_name.eq_ignore_ascii_case(expected_bin)
                || file_name.eq_ignore_ascii_case("liquid-rust.exe")
                || file_name.eq_ignore_ascii_case("liquid-rust")
            {
                let mut bin = Vec::new();
                file.read_to_end(&mut bin)
                    .map_err(|e| format!("Failed extracting binary from zip: {}", e))?;
                return Ok(bin);
            }
        }
        return Err(format!(
            "Zip archive '{}' did not contain '{}'",
            asset_name, expected_bin
        ));
    }

    Ok(raw_bytes.to_vec())
}

fn validate_executable_header(bin: &[u8]) -> Result<(), String> {
    if bin.len() < 64 * 1024 {
        return Err("Extracted executable is too small to be valid".to_string());
    }

    #[cfg(target_os = "linux")]
    {
        if !bin.starts_with(b"\x7fELF") {
            return Err("Downloaded binary does not have a valid Linux ELF header".to_string());
        }
    }

    #[cfg(target_os = "windows")]
    {
        if !bin.starts_with(b"MZ") {
            return Err("Downloaded binary does not have a valid Windows PE (MZ) header".to_string());
        }
    }

    #[cfg(target_os = "macos")]
    {
        let magic = [bin[0], bin[1], bin[2], bin[3]];
        let valid_macho = matches!(
            magic,
            [0xCF, 0xFA, 0xED, 0xFE]
                | [0xFE, 0xED, 0xFA, 0xCF]
                | [0xCA, 0xFE, 0xBA, 0xBE]
                | [0xBE, 0xBA, 0xFE, 0xCA]
        );
        if !valid_macho {
            return Err("Downloaded binary does not have a valid macOS Mach-O header".to_string());
        }
    }

    Ok(())
}

fn extract_and_install_binary(asset_name: &str, raw_bytes: &[u8]) -> Result<PathBuf, String> {
    let bin = extract_binary_from_archive(asset_name, raw_bytes)?;
    validate_executable_header(&bin)?;
    let target_exe = resolve_target_executable_path()?;
    install_binary_bytes(&bin, &target_exe)
}

fn install_binary_bytes(bin: &[u8], target_exe: &Path) -> Result<PathBuf, String> {
    let parent = target_exe
        .parent()
        .ok_or_else(|| format!("Invalid target path '{}'", target_exe.display()))?;

    let tmp_path = parent.join(".liquid-rust.update.tmp");
    match fs::File::create(&tmp_path) {
        Ok(mut f) => {
            f.write_all(bin)
                .map_err(|e| format!("Failed writing temporary update binary: {}", e))?;
            f.sync_all()
                .map_err(|e| format!("Failed syncing update binary to disk: {}", e))?;
            drop(f);

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o755))
                    .map_err(|e| format!("Failed setting executable permissions: {}", e))?;
            }

            #[cfg(windows)]
            {
                let old_path = target_exe.with_extension("exe.old");
                let _ = fs::remove_file(&old_path);
                if target_exe.exists() {
                    fs::rename(target_exe, &old_path).map_err(|e| {
                        let _ = fs::remove_file(&tmp_path);
                        format!("Failed moving active exe aside: {}", e)
                    })?;
                }
                if let Err(e) = fs::rename(&tmp_path, target_exe) {
                    let _ = fs::rename(&old_path, target_exe);
                    return Err(format!("Failed placing updated exe: {}", e));
                }
            }

            #[cfg(not(windows))]
            {
                fs::rename(&tmp_path, target_exe).map_err(|e| {
                    let _ = fs::remove_file(&tmp_path);
                    format!("Failed replacing '{}': {}", target_exe.display(), e)
                })?;
            }

            Ok(target_exe.to_path_buf())
        }
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            install_elevated_or_user_fallback(bin, target_exe)
        }
        Err(e) => Err(format!(
            "Cannot write update into '{}': {}",
            parent.display(),
            e
        )),
    }
}

#[cfg(unix)]
fn install_elevated_or_user_fallback(bin: &[u8], target_exe: &Path) -> Result<PathBuf, String> {
    use std::io::IsTerminal;
    use std::os::unix::fs::PermissionsExt;

    let stage_path = std::env::temp_dir().join(format!("liquid-rust-update-{}", std::process::id()));
    {
        let mut f = fs::File::create(&stage_path)
            .map_err(|e| format!("Failed staging update in temp dir: {}", e))?;
        f.write_all(bin)
            .map_err(|e| format!("Failed writing staged update: {}", e))?;
        f.sync_all()
            .map_err(|e| format!("Failed syncing staged update: {}", e))?;
    }
    let _ = fs::set_permissions(&stage_path, fs::Permissions::from_mode(0o755));

    // Atomic stage + mv avoids ETXTBSY ("Text file busy") when replacing a running /bin/liquid-rust binary
    let atomic_script = r#"install -m 0755 "$1" "$2.new" && mv -f "$2.new" "$2""#;

    // 1. If running in an interactive terminal (e.g. `liquid-rust --update`), prefer `sudo`
    if std::io::stdin().is_terminal() {
        if let Ok(status) = Command::new("sudo")
            .arg("sh")
            .arg("-c")
            .arg(atomic_script)
            .arg("--")
            .arg(&stage_path)
            .arg(target_exe)
            .status()
        {
            if status.success() {
                let _ = fs::remove_file(&stage_path);
                return Ok(target_exe.to_path_buf());
            }
        }
    }

    // 2. Native desktop Polkit authentication (`pkexec`) for GUI updates of `/bin/liquid-rust` on Linux
    #[cfg(not(target_os = "macos"))]
    if let Ok(status) = Command::new("pkexec")
        .arg("sh")
        .arg("-c")
        .arg(atomic_script)
        .arg("--")
        .arg(&stage_path)
        .arg(target_exe)
        .status()
    {
        if status.success() {
            let _ = fs::remove_file(&stage_path);
            return Ok(target_exe.to_path_buf());
        }
    }

    // 2b. Native macOS GUI authentication (`osascript` Touch ID / Admin prompt) for `/usr/local/bin/liquid-rust`
    #[cfg(target_os = "macos")]
    {
        let esc = |s: &str| s.replace('\'', "'\\''");
        let stage_s = esc(&stage_path.to_string_lossy());
        let target_s = esc(&target_exe.to_string_lossy());
        let shell_cmd = format!(
            "install -m 0755 '{}' '{}.new' && mv -f '{}.new' '{}'",
            stage_s, target_s, target_s, target_s
        );
        let osa_expr = format!(
            "do shell script \"{}\" with administrator privileges",
            shell_cmd.replace('\\', "\\\\").replace('"', "\\\"")
        );
        if let Ok(status) = Command::new("/usr/bin/osascript")
            .arg("-e")
            .arg(&osa_expr)
            .status()
        {
            if status.success() {
                let _ = fs::remove_file(&stage_path);
                return Ok(target_exe.to_path_buf());
            }
        }
    }

    // 3. Fallback: install into `~/.local/bin/liquid-rust` if elevation is unavailable
    if let Some(home) = std::env::var_os("HOME") {
        let user_bin_dir = PathBuf::from(home).join(".local").join("bin");
        if fs::create_dir_all(&user_bin_dir).is_ok() {
            let user_exe = user_bin_dir.join("liquid-rust");
            let user_tmp = user_bin_dir.join(".liquid-rust.update.tmp");
            if fs::copy(&stage_path, &user_tmp).is_ok() {
                let _ = fs::set_permissions(&user_tmp, fs::Permissions::from_mode(0o755));
                if fs::rename(&user_tmp, &user_exe).is_ok() {
                    let _ = fs::remove_file(&stage_path);
                    return Ok(user_exe);
                }
            }
        }
    }

    let _ = fs::remove_file(&stage_path);
    Err(format!(
        "Permission denied updating '{}' (and pkexec/sudo elevation was cancelled or unavailable)",
        target_exe.display()
    ))
}

#[cfg(not(unix))]
fn install_elevated_or_user_fallback(_bin: &[u8], target_exe: &Path) -> Result<PathBuf, String> {
    Err(format!(
        "Permission denied updating '{}'. Please run as Administrator.",
        target_exe.display()
    ))
}

/// Spawn the newly installed executable with the original CLI arguments (excluding updater flags)
/// and terminate the current process immediately.
pub fn restart_into_updated_binary(target_exe: &Path) -> Result<(), String> {
    let mut forwarded_args = Vec::new();
    for arg in std::env::args_os().skip(1) {
        let s = arg.to_string_lossy();
        if matches!(
            s.as_ref(),
            "--update" | "--self-update" | "--check-update" | "--force-update"
        ) {
            continue;
        }
        forwarded_args.push(arg);
    }

    Command::new(target_exe)
        .args(&forwarded_args)
        .spawn()
        .map_err(|e| format!("Failed to relaunch '{}': {}", target_exe.display(), e))?;

    std::process::exit(0);
}

/// Format byte sizes cleanly (e.g. `3.4 MB`, `196 KB`).
pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// CLI `--check-update` entrypoint.
pub fn run_cli_check_update() -> i32 {
    println!("Checking GitHub ({}) for updates...", GITHUB_REPO);
    println!("  Installed version : v{}", CURRENT_VERSION);
    match fetch_latest_release() {
        Ok(info) => {
            println!("  Latest release    : {} ({})", info.tag_name, info.title);
            println!(
                "  Platform package  : {} ({})",
                info.asset_name,
                format_bytes(info.asset_size)
            );
            if info.is_newer {
                println!(
                    "\n⬆ Update available! Run `liquid-rust --update` to download and apply {}.",
                    info.tag_name
                );
            } else {
                println!("\n✔ LiquidRust is up to date (v{}).", CURRENT_VERSION);
            }
            0
        }
        Err(e) => {
            eprintln!("Error checking for updates: {}", e);
            1
        }
    }
}

/// CLI `--update` entrypoint.
pub fn run_cli_self_update(force: bool, restart_after: bool) -> i32 {
    println!("Checking GitHub ({}) for updates...", GITHUB_REPO);
    let info = match fetch_latest_release() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Error checking for updates: {}", e);
            return 1;
        }
    };

    if !info.is_newer && !force {
        println!(
            "✔ LiquidRust is already on the latest version (v{}). Use `--update --force-update` to reinstall.",
            CURRENT_VERSION
        );
        return 0;
    }

    println!(
        "Downloading {} — {} ({})...",
        info.tag_name,
        info.asset_name,
        format_bytes(info.asset_size)
    );

    let mut last_pct = 999u64;
    let raw = match download_asset_bytes(&info.download_url, info.asset_size, |dl, tot| {
        if tot > 0 {
            let pct = (dl * 100) / tot;
            if pct != last_pct && (pct % 5 == 0 || dl == tot) {
                last_pct = pct;
                eprint!(
                    "\r  Progress: {:>3}% ({} / {})   ",
                    pct,
                    format_bytes(dl),
                    format_bytes(tot)
                );
                let _ = std::io::stderr().flush();
            }
        }
    }) {
        Ok(b) => {
            eprintln!();
            b
        }
        Err(e) => {
            eprintln!("\nDownload failed: {}", e);
            return 1;
        }
    };

    println!("Applying update...");
    match extract_and_install_binary(&info.asset_name, &raw) {
        Ok(installed_path) => {
            println!(
                "✔ Successfully updated LiquidRust to {} at {}",
                info.tag_name,
                installed_path.display()
            );
            if restart_after {
                println!("Restarting LiquidRust...");
                if let Err(e) = restart_into_updated_binary(&installed_path) {
                    eprintln!("{}", e);
                    return 1;
                }
            }
            0
        }
        Err(e) => {
            eprintln!("Failed to install update: {}", e);
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semver_comparison() {
        assert_eq!(parse_semver("v0.3.2"), Some((0, 3, 2)));
        assert_eq!(parse_semver("0.10.1-rc1"), Some((0, 10, 1)));
        assert_eq!(compare_versions("v0.3.3", "0.3.2"), Ordering::Greater);
        assert_eq!(compare_versions("v0.3.2", "0.3.2"), Ordering::Equal);
        assert_eq!(compare_versions("v0.3.1", "0.3.2"), Ordering::Less);
        assert_eq!(compare_versions("v1.0.0", "0.9.9"), Ordering::Greater);
    }
}
