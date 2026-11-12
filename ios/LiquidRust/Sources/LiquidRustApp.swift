import SwiftUI
import UIKit

@main
struct LiquidRustApp: App {

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
