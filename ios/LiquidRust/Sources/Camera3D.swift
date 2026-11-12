import Foundation
import simd

final class Camera3D {

    private let fovYDeg: Float = 44.0

    private(set) var viewMatrix: simd_float4x4 = matrix_identity_float4x4
    private(set) var projMatrix: simd_float4x4 = matrix_identity_float4x4
    private(set) var viewProjMatrix: simd_float4x4 = matrix_identity_float4x4

    private var boundMin = SIMD3<Float>(-0.60, 0.00, -0.05)
    private var boundMax = SIMD3<Float>( 0.60, 1.86,  0.05)

    private var target = SIMD3<Float>(0.0, 0.90, 0.0)
    private var yaw: Float = 0.56
    private var pitch: Float = 0.35
    private var distance: Float = 3.20

    private var viewportWidth: Float = 1179.0
    private var viewportHeight: Float = 2556.0
    private var pendingAutoFit: Bool = true

    func setChamberBounds(min bMin: SIMD3<Float>, max bMax: SIMD3<Float>) {
        boundMin = bMin
        boundMax = bMax
    }

    func setViewport(width: Float, height: Float, is2D: Bool) {
        let w = max(1.0, width)
        let h = max(1.0, height)
        let changed = abs(w - viewportWidth) > 0.5 || abs(h - viewportHeight) > 0.5
        viewportWidth = w
        viewportHeight = h
        if changed || pendingAutoFit {
            reset(is2D: is2D)
        }
    }

    func reset(is2D: Bool) {
        let hSpan = boundMax.y - boundMin.y
        target = SIMD3<Float>(
            (boundMin.x + boundMax.x) * 0.5,
            (boundMin.y + boundMax.y) * 0.5 - (is2D ? hSpan * 0.015 : 0.0),
            (boundMin.z + boundMax.z) * 0.5
        )
        if is2D {
            yaw = 0.0
            pitch = 0.0
        } else {
            yaw = 0.58
            pitch = 0.35
        }
        distance = computeIdealDistance(is2D: is2D)
        pendingAutoFit = false
    }

    private func computeIdealDistance(is2D: Bool) -> Float {
        let aspect = max(0.2, viewportWidth / max(1.0, viewportHeight))
        let tanHalfY = tan((fovYDeg * 0.5) * .pi / 180.0)
        let tanHalfX = max(0.12, tanHalfY * aspect)

        if is2D {
            let wPad = (boundMax.x - boundMin.x) * 1.10
            let hPad = (boundMax.y - boundMin.y) * 1.28
            return max((wPad * 0.5) / tanHalfX, (hPad * 0.5) / tanHalfY)
        } else {
            let spanXZ = hypot(boundMax.x - boundMin.x, boundMax.z - boundMin.z) * 1.12
            let spanY = (boundMax.y - boundMin.y) * 1.38
            return max((spanXZ * 0.5) / tanHalfX, (spanY * 0.5) / tanHalfY)
        }
    }

    func orbit(dYaw: Float, dPitch: Float) {
        yaw += dYaw
        pitch = min(1.35, max(-1.35, pitch + dPitch))
    }

    func pan(dxPx: Float, dyPx: Float) {
        let scale = distance / max(320.0, min(viewportWidth, viewportHeight))
        let cosYaw = cos(yaw)
        let sinYaw = sin(yaw)
        target.x -= dxPx * scale * cosYaw
        target.z += dxPx * scale * sinYaw
        target.y += dyPx * scale
    }

    func zoom(by factor: Float) {
        guard factor > 0.01 else { return }
        distance = min(8.50, max(0.80, distance / factor))
    }

    func updateMatrices(is2D: Bool) {
        let aspect = max(0.1, viewportWidth / max(1.0, viewportHeight))
        projMatrix = Self.perspectiveRightHandMetal(
            fovyRadians: fovYDeg * .pi / 180.0,
            aspect: aspect,
            nearZ: 0.08,
            farZ: 32.0
        )

        let eye: SIMD3<Float>
        if is2D {
            eye = SIMD3<Float>(target.x, target.y, distance)
        } else {
            let cosP = cos(pitch)
            eye = SIMD3<Float>(
                target.x + distance * cosP * sin(yaw),
                target.y + distance * sin(pitch),
                target.z + distance * cosP * cos(yaw)
            )
        }

        viewMatrix = Self.lookAtRightHand(eye: eye, center: target, up: SIMD3<Float>(0, 1, 0))
        viewProjMatrix = projMatrix * viewMatrix
    }

    func computeTouchRay(xPx: Float, yPx: Float, viewWidth: Float, viewHeight: Float, is2D: Bool, cfg: SimulationConfig) {
        updateMatrices(is2D: is2D)
        let invVP = viewProjMatrix.inverse

        let ndcX = (xPx / max(1.0, viewWidth)) * 2.0 - 1.0
        let ndcY = 1.0 - (yPx / max(1.0, viewHeight)) * 2.0

        // Metal NDC depth spans [0, 1]
        let nearClip = SIMD4<Float>(ndcX, ndcY, 0.0, 1.0)
        let farClip  = SIMD4<Float>(ndcX, ndcY, 1.0, 1.0)

        let nearW = invVP * nearClip
        let farW  = invVP * farClip

        let pNear = nearW.xyz / (abs(nearW.w) > 1e-6 ? nearW.w : 1.0)
        let pFar  = farW.xyz  / (abs(farW.w)  > 1e-6 ? farW.w  : 1.0)

        let dir = pFar - pNear
        let len = simd_length(dir)
        if len > 1e-6 {
            cfg.rayOrig = pNear
            cfg.rayDir = dir / len
        }
    }

    // Metal right-handed [0, 1] clip-space perspective matrix
    private static func perspectiveRightHandMetal(fovyRadians: Float, aspect: Float, nearZ: Float, farZ: Float) -> simd_float4x4 {
        let ys = 1.0 / tan(fovyRadians * 0.5)
        let xs = ys / aspect
        let zs = farZ / (nearZ - farZ)
        return simd_float4x4(columns: (
            SIMD4<Float>(xs,  0,  0,   0),
            SIMD4<Float>( 0, ys,  0,   0),
            SIMD4<Float>( 0,  0, zs,  -1),
            SIMD4<Float>( 0,  0, zs * nearZ, 0)
        ))
    }

    private static func lookAtRightHand(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let z = simd_normalize(eye - center)
        let x = simd_normalize(simd_cross(up, z))
        let y = simd_cross(z, x)
        return simd_float4x4(columns: (
            SIMD4<Float>(x.x, y.x, z.x, 0),
            SIMD4<Float>(x.y, y.y, z.y, 0),
            SIMD4<Float>(x.z, y.z, z.z, 0),
            SIMD4<Float>(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1)
        ))
    }
}

private extension SIMD4 where Scalar == Float {
    var xyz: SIMD3<Float> { SIMD3<Float>(x, y, z) }
}
