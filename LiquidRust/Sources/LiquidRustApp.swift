import SwiftUI
import UIKit

final class AppDelegate: NSObject, UIApplicationDelegate {
    static var orientationLock: UIInterfaceOrientationMask = .allButUpsideDown

    func application(
        _ application: UIApplication,
        supportedInterfaceOrientationsFor window: UIWindow?
    ) -> UIInterfaceOrientationMask {
        Self.orientationLock
    }
}

enum OrientationController {
    @MainActor
    static func setLocked(_ locked: Bool) {
        let mask: UIInterfaceOrientationMask
        if locked {
            let current = UIApplication.shared.connectedScenes
                .compactMap { $0 as? UIWindowScene }
                .first?.interfaceOrientation ?? .portrait
            switch current {
            case .landscapeLeft:
                mask = .landscapeLeft
            case .landscapeRight:
                mask = .landscapeRight
            case .portraitUpsideDown:
                mask = .portraitUpsideDown
            default:
                mask = .portrait
            }
        } else {
            mask = .allButUpsideDown
        }

        AppDelegate.orientationLock = mask

        for scene in UIApplication.shared.connectedScenes {
            guard let windowScene = scene as? UIWindowScene else { continue }
            for window in windowScene.windows {
                window.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
            }
            if #available(iOS 16.0, *) {
                windowScene.requestGeometryUpdate(.iOS(interfaceOrientations: mask)) { _ in }
            }
        }
    }
}

@main
struct LiquidRustApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    init() {
        UIApplication.shared.isIdleTimerDisabled = true
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .statusBarHidden(false)
        }
    }
}
