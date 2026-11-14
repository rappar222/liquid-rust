import Foundation
import SwiftUI
import UIKit

struct GitHubReleaseInfo: Equatable {
    let tagName: String
    let version: String
    let title: String
    let notes: String
    let htmlURL: String
    let ipaDownloadURL: String
    let ipaSizeBytes: Int64
    let isNewer: Bool

    var formattedSize: String {
        guard ipaSizeBytes > 0 else { return "IPA" }
        let mb = Double(ipaSizeBytes) / (1024.0 * 1024.0)
        if mb >= 1.0 {
            return String(format: "%.1f MB", mb)
        } else {
            return String(format: "%.0f KB", Double(ipaSizeBytes) / 1024.0)
        }
    }
}

@MainActor
final class AppUpdater: ObservableObject {
    static let latestReleaseAPI = "https://api.github.com/repos/rappar222/liquid-rust/releases/latest"
    static let fallbackIpaURL = "https://github.com/rappar222/liquid-rust/releases/latest/download/liquid-rust-ios.ipa"

    @Published var isChecking: Bool = false
    @Published var latestRelease: GitHubReleaseInfo? = nil
    @Published var errorMessage: String? = nil
    @Published var showUpdateSheet: Bool = false

    private var hasCheckedStartup: Bool = false

    var currentVersion: String {
        (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "0.3.11"
    }

    var hasUpdateAvailable: Bool {
        latestRelease?.isNewer == true
    }

    func checkOnLaunchIfNeeded() async {
        guard !hasCheckedStartup else { return }
        hasCheckedStartup = true
        await checkForUpdates(manual: false)
    }

    func checkForUpdates(manual: Bool) async {
        guard !isChecking else { return }
        isChecking = true
        errorMessage = nil
        if manual {
            showUpdateSheet = true
        }

        defer { isChecking = false }

        guard let url = URL(string: Self.latestReleaseAPI) else {
            errorMessage = "Invalid GitHub release URL."
            return
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("LiquidRust-iOS/\(currentVersion)", forHTTPHeaderField: "User-Agent")

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                errorMessage = "GitHub API returned HTTP \(http.statusCode)"
                return
            }

            let decoded = try JSONDecoder().decode(GitHubReleasePayload.self, from: data)
            let remoteVer = Self.normalizeVersion(decoded.tag_name)
            let newer = Self.compareVersions(remoteVer, currentVersion) == .orderedDescending

            let ipaAsset = decoded.assets.first {
                $0.name.lowercased() == "liquid-rust-ios.ipa" || $0.name.lowercased().hasSuffix(".ipa")
            }

            let info = GitHubReleaseInfo(
                tagName: decoded.tag_name,
                version: remoteVer,
                title: (decoded.name?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false)
                    ? decoded.name!.trimmingCharacters(in: .whitespacesAndNewlines)
                    : "LiquidRust \(decoded.tag_name)",
                notes: (decoded.body ?? "").trimmingCharacters(in: .whitespacesAndNewlines),
                htmlURL: decoded.html_url,
                ipaDownloadURL: ipaAsset?.browser_download_url ?? Self.fallbackIpaURL,
                ipaSizeBytes: ipaAsset?.size ?? 0,
                isNewer: newer
            )

            latestRelease = info
            if newer && !manual {
                showUpdateSheet = true
            }
        } catch {
            if manual {
                errorMessage = "Could not reach GitHub: \(error.localizedDescription)"
            }
        }
    }

    func installViaSideStore() {
        let ipa = latestRelease?.ipaDownloadURL ?? Self.fallbackIpaURL
        guard let encoded = ipa.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "sidestore://install?url=\(encoded)") else { return }
        UIApplication.shared.open(url) { opened in
            if !opened {
                self.downloadIpaDirectly()
            }
        }
    }

    func installViaAltStore() {
        let ipa = latestRelease?.ipaDownloadURL ?? Self.fallbackIpaURL
        guard let encoded = ipa.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "altstore://install?url=\(encoded)") else { return }
        UIApplication.shared.open(url) { opened in
            if !opened {
                self.downloadIpaDirectly()
            }
        }
    }

    func downloadIpaDirectly() {
        let ipa = latestRelease?.ipaDownloadURL ?? Self.fallbackIpaURL
        if let url = URL(string: ipa) {
            UIApplication.shared.open(url)
        }
    }

    static func normalizeVersion(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("v") || s.hasPrefix("V") {
            s.removeFirst()
        }
        if let idx = s.firstIndex(where: { $0 == "-" || $0 == "+" }) {
            s = String(s[..<idx])
        }
        return s
    }

    static func compareVersions(_ a: String, _ b: String) -> ComparisonResult {
        let pa = normalizeVersion(a).split(separator: ".").map { Int($0) ?? 0 }
        let pb = normalizeVersion(b).split(separator: ".").map { Int($0) ?? 0 }
        let count = max(pa.count, pb.count, 3)
        for i in 0..<count {
            let va = i < pa.count ? pa[i] : 0
            let vb = i < pb.count ? pb[i] : 0
            if va > vb { return .orderedDescending }
            if va < vb { return .orderedAscending }
        }
        return .orderedSame
    }
}

private struct GitHubReleasePayload: Decodable {
    let tag_name: String
    let name: String?
    let body: String?
    let html_url: String
    let assets: [GitHubAssetPayload]
}

private struct GitHubAssetPayload: Decodable {
    let name: String
    let browser_download_url: String
    let size: Int64
}
