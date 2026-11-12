import Foundation
import Metal
import MetalKit
import CoreMotion
import SwiftUI
import UIKit
import simd

private struct SimUniformsGPU {
    var bounds_min: SIMD4<Float>
    var bounds_max: SIMD4<Float>
    var gravity_dt: SIMD4<Float>
    var grid_dim:   SIMD4<Int32>
    var touch_orig: SIMD4<Float>
    var touch_dir:  SIMD4<Float>
    var obstacle:   SIMD4<Float>
    var kernel_p:   SIMD4<Float>
    var factors:    SIMD4<Float>
    var solver:     SIMD4<Float>
    var fluid:      SIMD4<Float>
    var counts:     SIMD4<Int32>
    var extra:      SIMD4<Float>
}

private struct RenderUniformsGPU {
    var view:         simd_float4x4
    var proj:         simd_float4x4
    var view_proj:    simd_float4x4
    var params:       SIMD4<Float>
    var modes:        SIMD4<Int32>
    var light_foam:   SIMD4<Float>
    var shallow_refr: SIMD4<Float>
    var deep_pad:     SIMD4<Float>
    var optics:       SIMD4<Float>
    var line_color:   SIMD4<Float>
}

// Interactive Metal Touch View forwarding multi-touch gestures & 120Hz ProMotion
final class InteractiveMTKView: MTKView {
    weak var engine: MetalFluidEngine?

    override init(frame frameRect: CGRect, device: MTLDevice?) {
        super.init(frame: frameRect, device: device)
        setupGestures()
    }

    required init(coder: NSCoder) {
        super.init(coder: coder)
        setupGestures()
    }

    private func setupGestures() {
        isMultipleTouchEnabled = true

        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        pinch.cancelsTouchesInView = false
        addGestureRecognizer(pinch)

        let twoPan = UIPanGestureRecognizer(target: self, action: #selector(handleTwoFingerPan(_:)))
        twoPan.minimumNumberOfTouches = 2
        twoPan.maximumNumberOfTouches = 2
        twoPan.cancelsTouchesInView = false
        addGestureRecognizer(twoPan)
    }

    @objc private func handlePinch(_ g: UIPinchGestureRecognizer) {
        guard let engine = engine else { return }
        if g.state == .changed {
            engine.cfg.activeTouchForceMode = 0
            engine.camera.zoom(by: Float(g.scale))
            g.scale = 1.0
        }
    }

    private var lastTwoPan: CGPoint = .zero
    @objc private func handleTwoFingerPan(_ g: UIPanGestureRecognizer) {
        guard let engine = engine else { return }
        let pt = g.translation(in: self)
        if g.state == .began {
            lastTwoPan = pt
            engine.cfg.activeTouchForceMode = 0
        } else if g.state == .changed {
            let scale = Float(contentScaleFactor)
            let dx = Float(pt.x - lastTwoPan.x) * scale
            let dy = Float(pt.y - lastTwoPan.y) * scale
            lastTwoPan = pt
            engine.camera.pan(dxPx: dx, dyPx: dy)
        }
    }

    private var lastSinglePt: CGPoint = .zero

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let engine = engine else { return }
        let allCount = event?.allTouches?.count ?? touches.count
        if allCount >= 2 {
            engine.cfg.activeTouchForceMode = 0
            return
        }
        guard let t = touches.first else { return }
        let loc = t.location(in: self)
        lastSinglePt = loc
        applySingleTouch(loc: loc, prev: loc, engine: engine)
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let engine = engine else { return }
        let allCount = event?.allTouches?.count ?? touches.count
        if allCount >= 2 {
            engine.cfg.activeTouchForceMode = 0
            return
        }
        guard let t = touches.first else { return }
        let loc = t.location(in: self)
        applySingleTouch(loc: loc, prev: lastSinglePt, engine: engine)
        lastSinglePt = loc
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        engine?.cfg.activeTouchForceMode = 0
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        engine?.cfg.activeTouchForceMode = 0
    }

    private func applySingleTouch(loc: CGPoint, prev: CGPoint, engine: MetalFluidEngine) {
        let w = Float(bounds.width)
        let h = Float(bounds.height)
        let tool = engine.cfg.touchTool
        if tool == 0 {
            engine.cfg.activeTouchForceMode = 1
            engine.camera.computeTouchRay(
                xPx: Float(loc.x), yPx: Float(loc.y),
                viewWidth: w, viewHeight: h,
                is2D: engine.cfg.is2D, cfg: engine.cfg
            )
        } else if tool == 1 {
            engine.cfg.activeTouchForceMode = -1
            engine.camera.computeTouchRay(
                xPx: Float(loc.x), yPx: Float(loc.y),
                viewWidth: w, viewHeight: h,
                is2D: engine.cfg.is2D, cfg: engine.cfg
            )
        } else {
            engine.cfg.activeTouchForceMode = 0
            let scale = Float(contentScaleFactor)
            let dx = Float(loc.x - prev.x) * scale
            let dy = Float(loc.y - prev.y) * scale
            if engine.cfg.is2D {
                engine.camera.pan(dxPx: dx, dyPx: dy)
            } else {
                engine.camera.orbit(dYaw: -dx * 0.0045, dPitch: dy * 0.0045)
            }
        }
    }
}

final class MetalFluidEngine: NSObject, MTKViewDelegate {

    let cfg: SimulationConfig
    let camera = Camera3D()

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?

    // 8 Compute Pipelines
    private var pipePredict: MTLComputePipelineState?
    private var pipeClearGrid: MTLComputePipelineState?
    private var pipeInsertGrid: MTLComputePipelineState?
    private var pipeDensity: MTLComputePipelineState?
    private var pipeSolveCollide: MTLComputePipelineState?
    private var pipeVelocityXsph: MTLComputePipelineState?
    private var pipeCurl: MTLComputePipelineState?
    private var pipeVorticity: MTLComputePipelineState?

    // Render Pipelines & Depth States
    private var pipeParticle2D: MTLRenderPipelineState?
    private var pipeParticle3D: MTLRenderPipelineState?
    private var pipeLine: MTLRenderPipelineState?
    private var depthDisabled: MTLDepthStencilState?
    private var depthLessEqual: MTLDepthStencilState?

    // Unified Apple Silicon GPU Buffers
    private var bufPos: MTLBuffer?
    private var bufVel: MTLBuffer?
    private var bufPredA: MTLBuffer?
    private var bufPredB: MTLBuffer?
    private var bufGridHeads: MTLBuffer?
    private var bufGridNext: MTLBuffer?
    private var bufWire: MTLBuffer?

    private var allocatedParticles: Int = 0
    private var allocatedCells: Int = 0

    // Solver domain state
    private var minBounds = SIMD3<Float>(-0.60, 0.00, -0.05)
    private var maxBounds = SIMD3<Float>( 0.60, 1.86,  0.05)
    private var particleCount: Int = 0
    private var active2D: Bool = true
    private var spacing: Float = 0.028
    private var particleRadius: Float = 0.0145
    private var h: Float = 0.0574
    private var h2: Float = 0.0033
    private var poly6Factor: Float = 1.0
    private var spikyFactor: Float = 1.0
    private var wQ: Float = 1.0
    private var selfDensity: Float = 1.0
    private var restDensity: Float = 1000.0
    private var invRestDensity: Float = 0.001
    private var gridDim = SIMD3<Int32>(32, 22, 1)
    private var totalCells: Int = 704
    private var solverIterations: Int = 3
    private let substeps: Int = 2

    private let lightDir = simd_normalize(SIMD3<Float>(0.42, 0.82, 0.48))
    private var wireScratch: [SIMD4<Float>] = Array(repeating: .zero, count: 512)

    // Motion & Telemetry
    private let motionManager = CMMotionManager()
    private var motionActive: Bool = false
    private var animTime: Float = 0.0
    private var lastFrameTime: CFTimeInterval = CACurrentMediaTime()
    private var telemetryStart: CFTimeInterval = CACurrentMediaTime()
    private var frameCounter: Int = 0
    private var accumStepMs: Float = 0.0
    private var stepCounter: Int = 0

    init(cfg: SimulationConfig) {
        self.cfg = cfg
        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()
        super.init()

        if let dev = dev {
            buildPipelines(device: dev)
            let rawName = dev.name
            DispatchQueue.main.async {
                cfg.gpuName = rawName
            }
        }
    }

    deinit {
        motionManager.stopDeviceMotionUpdates()
    }

    private func buildPipelines(device: MTLDevice) {
        guard let lib = device.makeDefaultLibrary() else { return }

        func makeComp(_ name: String) -> MTLComputePipelineState? {
            guard let fn = lib.makeFunction(name: name) else { return nil }
            return try? device.makeComputePipelineState(function: fn)
        }

        pipePredict      = makeComp("pbf_predict")
        pipeClearGrid    = makeComp("pbf_clear_grid")
        pipeInsertGrid   = makeComp("pbf_insert_grid")
        pipeDensity      = makeComp("pbf_density")
        pipeSolveCollide = makeComp("pbf_solve_collide")
        pipeVelocityXsph = makeComp("pbf_velocity_xsph")
        pipeCurl         = makeComp("pbf_curl")
        pipeVorticity    = makeComp("pbf_vorticity")

        if let vPart = lib.makeFunction(name: "particle_vert"),
           let f2d = lib.makeFunction(name: "particle_frag_2d"),
           let f3d = lib.makeFunction(name: "particle_frag_3d") {
            let desc2D = MTLRenderPipelineDescriptor()
            desc2D.vertexFunction = vPart
            desc2D.fragmentFunction = f2d
            desc2D.colorAttachments[0].pixelFormat = .bgra8Unorm
            desc2D.depthAttachmentPixelFormat = .depth32Float
            pipeParticle2D = try? device.makeRenderPipelineState(descriptor: desc2D)

            let desc3D = MTLRenderPipelineDescriptor()
            desc3D.vertexFunction = vPart
            desc3D.fragmentFunction = f3d
            desc3D.colorAttachments[0].pixelFormat = .bgra8Unorm
            desc3D.depthAttachmentPixelFormat = .depth32Float
            pipeParticle3D = try? device.makeRenderPipelineState(descriptor: desc3D)
        }

        if let vLine = lib.makeFunction(name: "line_vert"),
           let fLine = lib.makeFunction(name: "line_frag") {
            let lDesc = MTLRenderPipelineDescriptor()
            lDesc.vertexFunction = vLine
            lDesc.fragmentFunction = fLine
            lDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
            lDesc.colorAttachments[0].isBlendingEnabled = true
            lDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            lDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            lDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
            lDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            lDesc.depthAttachmentPixelFormat = .depth32Float
            pipeLine = try? device.makeRenderPipelineState(descriptor: lDesc)
        }

        let dOff = MTLDepthStencilDescriptor()
        dOff.depthCompareFunction = .always
        dOff.isDepthWriteEnabled = false
        depthDisabled = device.makeDepthStencilState(descriptor: dOff)

        let dOn = MTLDepthStencilDescriptor()
        dOn.depthCompareFunction = .lessEqual
        dOn.isDepthWriteEnabled = true
        depthLessEqual = device.makeDepthStencilState(descriptor: dOn)

        bufWire = device.makeBuffer(
            length: wireScratch.count * MemoryLayout<SIMD4<Float>>.stride,
            options: .storageModeShared
        )
    }

    private func syncMotionSensor() {
        let want = cfg.tiltGravityEnabled
        if want && !motionActive {
            if motionManager.isDeviceMotionAvailable {
                motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
                motionManager.startDeviceMotionUpdates()
                motionActive = true
            }
        } else if !want && motionActive {
            motionManager.stopDeviceMotionUpdates()
            motionActive = false
        }

        if motionActive, let g = motionManager.deviceMotion?.gravity {
            // CoreMotion gravity vector points toward Earth in device coordinates (x right, y up, z out of screen)
            let v = SIMD3<Float>(Float(g.x), Float(g.y), Float(g.z))
            let len = simd_length(v)
            if len > 0.15 {
                cfg.tiltDir = v / len
            }
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        let w = Float(max(1.0, size.width))
        let hPx = Float(max(1.0, size.height))
        let portraitNow = (hPx >= w)
        if portraitNow != cfg.isPortrait {
            cfg.isPortrait = portraitNow
            if cfg.is2D {
                cfg.reloadRequested = true
            }
        }
        camera.setViewport(width: w, height: hPx, is2D: cfg.is2D)
    }

    private func loadScene() {
        guard let device = device else { return }

        active2D = cfg.is2D
        let q = cfg.quality

        if active2D {
            if cfg.isPortrait {
                minBounds = SIMD3<Float>(-0.60, 0.00, -0.05)
                maxBounds = SIMD3<Float>( 0.60, 1.86,  0.05)
            } else {
                minBounds = SIMD3<Float>(-0.96, 0.00, -0.05)
                maxBounds = SIMD3<Float>( 0.96, 1.12,  0.05)
            }
            switch q {
            case 0:  spacing = 0.0280; solverIterations = 3
            case 1:  spacing = 0.0170; solverIterations = 3
            case 2:  spacing = 0.0120; solverIterations = 3
            case 3:  spacing = 0.0085; solverIterations = 4
            default: spacing = 0.0062; solverIterations = 4
            }
            h = spacing * 2.05
        } else {
            minBounds = SIMD3<Float>(-0.65, 0.00, -0.55)
            maxBounds = SIMD3<Float>( 0.65, 1.05,  0.55)
            switch q {
            case 0:  spacing = 0.0520; solverIterations = 3
            case 1:  spacing = 0.0360; solverIterations = 3
            case 2:  spacing = 0.0280; solverIterations = 3
            case 3:  spacing = 0.0225; solverIterations = 3
            default: spacing = 0.0185; solverIterations = 4
            }
            h = spacing * 1.95
        }

        particleRadius = spacing * 0.52
        h2 = h * h
        poly6Factor = Float(315.0 / (64.0 * Double.pi * pow(Double(h), 9.0)))
        spikyFactor = Float(45.0 / (Double.pi * pow(Double(h), 6.0)))
        selfDensity = poly6Factor * h2 * h2 * h2
        let qDist = 0.3 * h
        let diffQ = h2 - qDist * qDist
        wQ = poly6Factor * diffQ * diffQ * diffQ

        restDensity = computeLatticeRestDensity(is2D: active2D, h: h, h2: h2, poly6: poly6Factor, s: spacing)
        invRestDensity = 1.0 / restDensity

        let nx = max(4, Int32(ceil((maxBounds.x - minBounds.x) / h)) + 1)
        let ny = max(4, Int32(ceil((maxBounds.y - minBounds.y) / h)) + 1)
        let nz = active2D ? Int32(1) : max(4, Int32(ceil((maxBounds.z - minBounds.z) / h)) + 1)
        gridDim = SIMD3<Int32>(nx, ny, nz)
        totalCells = Int(nx * ny * nz)

        var initPos: [SIMD4<Float>] = []
        var initVel: [SIMD4<Float>] = []
        initPos.reserveCapacity(8192)
        initVel.reserveCapacity(8192)

        func addP(_ x: Float, _ y: Float, _ z: Float, _ vx: Float, _ vy: Float, _ vz: Float) {
            initPos.append(SIMD4<Float>(x, y, active2D ? 0.0 : z, restDensity))
            initVel.append(SIMD4<Float>(vx, vy, active2D ? 0.0 : vz, 0.0))
        }

        func spawnBlock(_ x0: Float, _ y0: Float, _ z0: Float, _ x1: Float, _ y1: Float, _ z1: Float, _ vx: Float, _ vy: Float, _ vz: Float) {
            var y = y0
            while y <= y1 {
                var x = x0
                while x <= x1 {
                    if active2D {
                        addP(x, y, 0, vx, vy, vz)
                    } else {
                        var z = z0
                        while z <= z1 {
                            addP(x, y, z, vx, vy, vz)
                            z += spacing
                        }
                    }
                    x += spacing
                }
                y += spacing
            }
        }

        func spawnSphere(_ cx: Float, _ cy: Float, _ cz: Float, _ rad: Float, _ vx: Float, _ vy: Float, _ vz: Float) {
            let r2 = rad * rad
            let steps = Int(ceil(rad / spacing))
            let zR = active2D ? 0 : steps
            for iz in -zR...zR {
                for iy in -steps...steps {
                    for ix in -steps...steps {
                        let ox = Float(ix) * spacing
                        let oy = Float(iy) * spacing
                        let oz = Float(iz) * spacing
                        if ox * ox + oy * oy + oz * oz <= r2 {
                            addP(cx + ox, cy + oy, cz + oz, vx, vy, vz)
                        }
                    }
                }
            }
        }

        let pad = particleRadius * 1.6
        let w = maxBounds.x - minBounds.x
        let hBox = maxBounds.y - minBounds.y
        let z0 = minBounds.z + pad
        let z1 = maxBounds.z - pad

        switch cfg.preset {
        case 1: // Double Dam
            let colW = w * 0.27
            let colH = hBox * 0.72
            spawnBlock(minBounds.x + pad, minBounds.y + pad, z0, minBounds.x + colW, minBounds.y + colH, z1, 0, 0, 0)
            spawnBlock(maxBounds.x - colW, minBounds.y + pad, z0, maxBounds.x - pad, minBounds.y + colH, z1, 0, 0, 0)
        case 2: // Droplet Splash
            spawnBlock(minBounds.x + pad, minBounds.y + pad, z0, maxBounds.x - pad, minBounds.y + hBox * 0.24, z1, 0, 0, 0)
            spawnSphere(0, minBounds.y + hBox * 0.76, 0, min(w, hBox) * 0.17, 0, -2.4, 0)
        case 3: // Obstacle Drop
            let bw = w * 0.56
            spawnBlock(-bw * 0.5, minBounds.y + hBox * 0.54, z0 * 0.75, bw * 0.5, minBounds.y + hBox * 0.92, z1 * 0.75, 0, -0.5, 0)
        case 4: // Vortex Swirl
            let rOuter = min(w, hBox) * 0.36
            let cy = minBounds.y + hBox * 0.52
            let steps = Int(ceil(rOuter / spacing))
            let zR = active2D ? 0 : Int(ceil(((z1 - z0) * 0.42) / spacing))
            for iz in -zR...zR {
                for iy in -steps...steps {
                    for ix in -steps...steps {
                        let ox = Float(ix) * spacing
                        let oy = Float(iy) * spacing
                        let oz = Float(iz) * spacing
                        let d2 = ox * ox + oy * oy
                        if d2 <= rOuter * rOuter {
                            let d = sqrt(d2) + 1e-4
                            let swirl = 3.2 * (d / rOuter)
                            if active2D {
                                addP(ox, cy + oy, 0, -oy / d * swirl, ox / d * swirl, 0)
                            } else {
                                let dr = sqrt(ox * ox + oz * oz) + 1e-4
                                addP(ox, cy + oy * 0.7, oz, -oz / dr * swirl, 0.2, ox / dr * swirl)
                            }
                        }
                    }
                }
            }
        case 5: // Zero-G Blobs
            let rad = min(w, hBox) * 0.20
            let cy = minBounds.y + hBox * 0.52
            spawnSphere(-w * 0.24, cy + 0.04, 0, rad,  1.45, -0.15, 0)
            spawnSphere( w * 0.24, cy - 0.04, 0, rad, -1.45,  0.15, 0)
        default: // Dam Break
            let tall = (hBox > w * 1.1)
            let damW = w * (tall ? 0.52 : 0.42)
            let damH = hBox * (tall ? 0.58 : 0.76)
            spawnBlock(minBounds.x + pad, minBounds.y + pad, z0, minBounds.x + pad + damW, minBounds.y + pad + damH, z1, 0, 0, 0)
        }

        particleCount = initPos.count
        let n = max(1, particleCount)

        if n > allocatedParticles {
            allocatedParticles = Int(Float(n) * 1.35) + 256
            let bytes = allocatedParticles * MemoryLayout<SIMD4<Float>>.stride
            bufPos      = device.makeBuffer(length: bytes, options: .storageModeShared)
            bufVel      = device.makeBuffer(length: bytes, options: .storageModeShared)
            bufPredA    = device.makeBuffer(length: bytes, options: .storageModeShared)
            bufPredB    = device.makeBuffer(length: bytes, options: .storageModeShared)
            bufGridNext = device.makeBuffer(length: allocatedParticles * MemoryLayout<Int32>.stride, options: .storageModeShared)
        }

        if totalCells > allocatedCells {
            allocatedCells = Int(Float(totalCells) * 1.25) + 512
            bufGridHeads = device.makeBuffer(length: allocatedCells * MemoryLayout<Int32>.stride, options: .storageModeShared)
        }

        if particleCount > 0 {
            let copyBytes = particleCount * MemoryLayout<SIMD4<Float>>.stride
            initPos.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    bufPos?.contents().copyMemory(from: base, byteCount: copyBytes)
                    bufPredA?.contents().copyMemory(from: base, byteCount: copyBytes)
                }
            }
            initVel.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    bufVel?.contents().copyMemory(from: base, byteCount: copyBytes)
                }
            }
        }

        camera.setChamberBounds(min: minBounds, max: maxBounds)
        let pCount = particleCount
        let hVal = h
        DispatchQueue.main.async { [weak self] in
            self?.cfg.particleCount = pCount
            self?.cfg.smoothingH = hVal
        }
    }

    private func computeLatticeRestDensity(is2D: Bool, h: Float, h2: Float, poly6: Float, s: Float) -> Float {
        let reach = Int(ceil(h / s)) + 1
        var sum: Float = 0.0
        let zR = is2D ? 0 : reach
        for dz in -zR...zR {
            for dy in -reach...reach {
                for dx in -reach...reach {
                    let rx = Float(dx) * s
                    let ry = Float(dy) * s
                    let rz = Float(dz) * s
                    let r2 = rx * rx + ry * ry + rz * rz
                    if r2 < h2 {
                        let diff = h2 - r2
                        sum += poly6 * diff * diff * diff
                    }
                }
            }
        }
        return max(sum * 0.96, 1.0)
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        var frameDt = Float(now - lastFrameTime)
        lastFrameTime = now
        if frameDt <= 0.0 || frameDt > 0.1 {
            frameDt = 1.0 / 60.0
        }

        syncMotionSensor()

        if cfg.reloadRequested {
            cfg.reloadRequested = false
            loadScene()
        }
        if cfg.resetCameraRequested {
            cfg.resetCameraRequested = false
            camera.reset(is2D: active2D)
        }

        guard let cmdQueue = commandQueue,
              let cmdBuf = cmdQueue.makeCommandBuffer(),
              let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }

        let doStep = cfg.isPlaying || cfg.stepOnceRequested
        if cfg.stepOnceRequested {
            cfg.stepOnceRequested = false
        }

        if doStep && particleCount > 0 {
            let t0 = CACurrentMediaTime()
            encodeComputeStep(cmdBuf: cmdBuf, frameDt: frameDt)
            let dtMs = Float((CACurrentMediaTime() - t0) * 1000.0)
            accumStepMs += dtMs
            stepCounter += 1
            animTime += min(frameDt, 1.0 / 45.0) * cfg.simSpeed
        }

        // Deep Obsidian Slate background complementing Liquid Glass UI
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.043, green: 0.062, blue: 0.094, alpha: 1.0)
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.depthAttachment.clearDepth = 1.0
        rpd.depthAttachment.loadAction = .clear
        rpd.depthAttachment.storeAction = .dontCare

        camera.updateMatrices(is2D: active2D)

        if let renderEnc = cmdBuf.makeRenderCommandEncoder(descriptor: rpd) {
            encodeWireframes(enc: renderEnc)
            encodeParticles(enc: renderEnc)
            renderEnc.endEncoding()
        }

        cmdBuf.present(drawable)
        cmdBuf.commit()

        frameCounter += 1
        let elapsed = now - telemetryStart
        if elapsed >= 0.45 {
            let fpsVal = Int(round(Double(frameCounter) / elapsed))
            let avgStep = stepCounter > 0 ? (accumStepMs / Float(stepCounter)) : 0.0
            let pCount = particleCount
            frameCounter = 0
            accumStepMs = 0.0
            stepCounter = 0
            telemetryStart = now

            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.cfg.fps = fpsVal
                self.cfg.particleCount = pCount
                self.cfg.stepMs = avgStep
            }
        }
    }

    private func encodeComputeStep(cmdBuf: MTLCommandBuffer, frameDt: Float) {
        guard let pipePredict = pipePredict,
              let pipeClearGrid = pipeClearGrid,
              let pipeInsertGrid = pipeInsertGrid,
              let pipeDensity = pipeDensity,
              let pipeSolveCollide = pipeSolveCollide,
              let pipeVelocityXsph = pipeVelocityXsph,
              var bPos = bufPos,
              let bVel = bufVel,
              var bPredA = bufPredA,
              let bPredB = bufPredB,
              let bHeads = bufGridHeads,
              let bNext = bufGridNext else { return }

        let totalDt = min(frameDt, 1.0 / 45.0) * cfg.simSpeed
        guard totalDt > 1e-6 else { return }

        let subDt = totalDt / Float(substeps)
        let dtFactor = max(0.02, min(1.0, subDt / (0.016 / Float(substeps))))
        let cfmScale = max(1.0, 1.0 / (dtFactor * dtFactor))

        let gMag = cfg.gravityMag
        let grav: SIMD3<Float>
        if cfg.tiltGravityEnabled {
            grav = SIMD3<Float>(
                cfg.tiltDir.x * gMag,
                cfg.tiltDir.y * gMag,
                active2D ? 0.0 : (cfg.tiltDir.z * gMag)
            )
        } else {
            grav = SIMD3<Float>(0.0, -gMag, 0.0)
        }

        let relaxEps = Float(1200.0 * pow(Double(0.075 / h), 2.0)) * cfmScale
        let maxDisp = max(particleRadius * 0.45, h * 0.25) * dtFactor
        let stCoeff = (cfg.surfaceTension * spacing * 0.085) / cfmScale

        var u = SimUniformsGPU(
            bounds_min: SIMD4<Float>(minBounds.x, minBounds.y, minBounds.z, h),
            bounds_max: SIMD4<Float>(maxBounds.x, maxBounds.y, maxBounds.z, particleRadius),
            gravity_dt: SIMD4<Float>(grav.x, grav.y, grav.z, subDt),
            grid_dim:   SIMD4<Int32>(gridDim.x, gridDim.y, gridDim.z, Int32(totalCells)),
            touch_orig: SIMD4<Float>(cfg.rayOrig.x, cfg.rayOrig.y, cfg.rayOrig.z, 26.0),
            touch_dir:  SIMD4<Float>(cfg.rayDir.x, cfg.rayDir.y, cfg.rayDir.z, Float(cfg.activeTouchForceMode)),
            obstacle:   SIMD4<Float>(cfg.obstacleX, cfg.obstacleY, 0.0, cfg.obstacleR),
            kernel_p:   SIMD4<Float>(h, h2, invRestDensity, restDensity),
            factors:    SIMD4<Float>(poly6Factor, spikyFactor, wQ, 0.00025),
            solver:     SIMD4<Float>(relaxEps, maxDisp, 0.12, 0.04),
            fluid:      SIMD4<Float>(cfg.viscosity, stCoeff, 8.5, cfg.obstacleEnabled ? 1.0 : 0.0),
            counts:     SIMD4<Int32>(Int32(particleCount), active2D ? 1 : 0, 0, 0),
            extra:      SIMD4<Float>(cfg.vorticity, spacing, cfmScale, selfDensity)
        )

        let tgSize = MTLSize(width: 64, height: 1, depth: 1)
        let pGroups = MTLSize(width: (particleCount + 63) / 64, height: 1, depth: 1)
        let cGroups = MTLSize(width: (totalCells + 63) / 64, height: 1, depth: 1)
        let doVort = (cfg.vorticity > 1e-4)

        guard let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        for _ in 0..<substeps {
            // 1. Predict
            enc.setComputePipelineState(pipePredict)
            enc.setBuffer(bPos, offset: 0, index: 0)
            enc.setBuffer(bVel, offset: 0, index: 1)
            enc.setBuffer(bPredA, offset: 0, index: 2)
            enc.setBytes(&u, length: MemoryLayout<SimUniformsGPU>.stride, index: 3)
            enc.dispatchThreadgroups(pGroups, threadsPerThreadgroup: tgSize)

            // 2. Clear Grid
            enc.setComputePipelineState(pipeClearGrid)
            enc.setBuffer(bHeads, offset: 0, index: 0)
            enc.setBytes(&u, length: MemoryLayout<SimUniformsGPU>.stride, index: 1)
            enc.dispatchThreadgroups(cGroups, threadsPerThreadgroup: tgSize)

            // 3. Insert Grid
            enc.setComputePipelineState(pipeInsertGrid)
            enc.setBuffer(bPredA, offset: 0, index: 0)
            enc.setBuffer(bHeads, offset: 0, index: 1)
            enc.setBuffer(bNext, offset: 0, index: 2)
            enc.setBytes(&u, length: MemoryLayout<SimUniformsGPU>.stride, index: 3)
            enc.dispatchThreadgroups(pGroups, threadsPerThreadgroup: tgSize)

            // 4 & 5. Jacobi Pressure + Cohesion + Obstacles
            for _ in 0..<solverIterations {
                enc.setComputePipelineState(pipeDensity)
                enc.setBuffer(bPredA, offset: 0, index: 0)
                enc.setBuffer(bHeads, offset: 0, index: 1)
                enc.setBuffer(bNext, offset: 0, index: 2)
                enc.setBuffer(bPredB, offset: 0, index: 3)
                enc.setBuffer(bPos, offset: 0, index: 4)
                enc.setBytes(&u, length: MemoryLayout<SimUniformsGPU>.stride, index: 5)
                enc.dispatchThreadgroups(pGroups, threadsPerThreadgroup: tgSize)

                enc.setComputePipelineState(pipeSolveCollide)
                enc.setBuffer(bPredB, offset: 0, index: 0)
                enc.setBuffer(bPos, offset: 0, index: 1)
                enc.setBuffer(bHeads, offset: 0, index: 2)
                enc.setBuffer(bNext, offset: 0, index: 3)
                enc.setBuffer(bPredA, offset: 0, index: 4)
                enc.setBytes(&u, length: MemoryLayout<SimUniformsGPU>.stride, index: 5)
                enc.dispatchThreadgroups(pGroups, threadsPerThreadgroup: tgSize)
            }

            // 6. Velocity + XSPH Viscosity
            enc.setComputePipelineState(pipeVelocityXsph)
            enc.setBuffer(bPredA, offset: 0, index: 0)
            enc.setBuffer(bPos, offset: 0, index: 1)
            enc.setBuffer(bHeads, offset: 0, index: 2)
            enc.setBuffer(bNext, offset: 0, index: 3)
            enc.setBuffer(bVel, offset: 0, index: 4)
            enc.setBytes(&u, length: MemoryLayout<SimUniformsGPU>.stride, index: 5)
            enc.dispatchThreadgroups(pGroups, threadsPerThreadgroup: tgSize)

            swap(&bPos, &bPredA)

            // 7 & 8. Vorticity Confinement
            if doVort, let pipeCurl = pipeCurl, let pipeVorticity = pipeVorticity {
                enc.setComputePipelineState(pipeCurl)
                enc.setBuffer(bPos, offset: 0, index: 0)
                enc.setBuffer(bVel, offset: 0, index: 1)
                enc.setBuffer(bHeads, offset: 0, index: 2)
                enc.setBuffer(bNext, offset: 0, index: 3)
                enc.setBuffer(bPredB, offset: 0, index: 4)
                enc.setBytes(&u, length: MemoryLayout<SimUniformsGPU>.stride, index: 5)
                enc.dispatchThreadgroups(pGroups, threadsPerThreadgroup: tgSize)

                enc.setComputePipelineState(pipeVorticity)
                enc.setBuffer(bPos, offset: 0, index: 0)
                enc.setBuffer(bPredB, offset: 0, index: 1)
                enc.setBuffer(bHeads, offset: 0, index: 2)
                enc.setBuffer(bNext, offset: 0, index: 3)
                enc.setBuffer(bVel, offset: 0, index: 4)
                enc.setBytes(&u, length: MemoryLayout<SimUniformsGPU>.stride, index: 5)
                enc.dispatchThreadgroups(pGroups, threadsPerThreadgroup: tgSize)
            }
        }

        enc.endEncoding()
        bufPos = bPos
        bufPredA = bPredA
    }

    private func makeRenderUniforms(lineColor: SIMD4<Float> = .zero) -> RenderUniformsGPU {
        RenderUniformsGPU(
            view: camera.viewMatrix,
            proj: camera.projMatrix,
            view_proj: camera.viewProjMatrix,
            params: SIMD4<Float>(particleRadius * 1.35, invRestDensity, cfg.texStrength, animTime),
            modes: SIMD4<Int32>(Int32(cfg.colorMode), Int32(cfg.colorMap), cfg.litSpheres ? 1 : 0, Int32(cfg.texMode)),
            light_foam: SIMD4<Float>(lightDir.x, lightDir.y, lightDir.z, cfg.foamAmount),
            shallow_refr: SIMD4<Float>(cfg.shallowRGB.x, cfg.shallowRGB.y, cfg.shallowRGB.z, cfg.refraction),
            deep_pad: SIMD4<Float>(cfg.deepRGB.x, cfg.deepRGB.y, cfg.deepRGB.z, 0.0),
            optics: SIMD4<Float>(cfg.ior, cfg.roughness, cfg.specular, cfg.metallic),
            line_color: lineColor
        )
    }

    private func encodeWireframes(enc: MTLRenderCommandEncoder) {
        guard let pipeLine = pipeLine, let bufWire = bufWire, let depthDisabled = depthDisabled else { return }

        var idx = 0
        func seg(_ a: SIMD3<Float>, _ b: SIMD3<Float>) {
            guard idx + 2 <= wireScratch.count else { return }
            wireScratch[idx]     = SIMD4<Float>(a.x, a.y, a.z, 1.0)
            wireScratch[idx + 1] = SIMD4<Float>(b.x, b.y, b.z, 1.0)
            idx += 2
        }

        let b0 = minBounds
        let b1 = maxBounds

        if !active2D {
            let yf = b0.y - 0.001
            let divs = 8
            for i in 1..<divs {
                let t = Float(i) / Float(divs)
                let gx = b0.x + (b1.x - b0.x) * t
                let gz = b0.z + (b1.z - b0.z) * t
                seg(SIMD3(gx, yf, b0.z), SIMD3(gx, yf, b1.z))
                seg(SIMD3(b0.x, yf, gz), SIMD3(b1.x, yf, gz))
            }
        }
        let gridVerts = idx

        if active2D {
            seg(SIMD3(b0.x, b0.y, 0), SIMD3(b1.x, b0.y, 0))
            seg(SIMD3(b1.x, b0.y, 0), SIMD3(b1.x, b1.y, 0))
            seg(SIMD3(b1.x, b1.y, 0), SIMD3(b0.x, b1.y, 0))
            seg(SIMD3(b0.x, b1.y, 0), SIMD3(b0.x, b0.y, 0))

            if cfg.obstacleEnabled {
                let ox = cfg.obstacleX, oy = cfg.obstacleY, r = cfg.obstacleR
                let segs = 36
                for i in 0..<segs {
                    let a0 = Float(i) * (.pi * 2.0) / Float(segs)
                    let a1 = Float(i + 1) * (.pi * 2.0) / Float(segs)
                    seg(SIMD3(ox + cos(a0) * r, oy + sin(a0) * r, 0),
                        SIMD3(ox + cos(a1) * r, oy + sin(a1) * r, 0))
                }
            }
        } else {
            seg(SIMD3(b0.x, b0.y, b0.z), SIMD3(b1.x, b0.y, b0.z))
            seg(SIMD3(b1.x, b0.y, b0.z), SIMD3(b1.x, b1.y, b0.z))
            seg(SIMD3(b1.x, b1.y, b0.z), SIMD3(b0.x, b1.y, b0.z))
            seg(SIMD3(b0.x, b1.y, b0.z), SIMD3(b0.x, b0.y, b0.z))

            seg(SIMD3(b0.x, b0.y, b1.z), SIMD3(b1.x, b0.y, b1.z))
            seg(SIMD3(b1.x, b0.y, b1.z), SIMD3(b1.x, b1.y, b1.z))
            seg(SIMD3(b1.x, b1.y, b1.z), SIMD3(b0.x, b1.y, b1.z))
            seg(SIMD3(b0.x, b1.y, b1.z), SIMD3(b0.x, b0.y, b1.z))

            seg(SIMD3(b0.x, b0.y, b0.z), SIMD3(b0.x, b0.y, b1.z))
            seg(SIMD3(b1.x, b0.y, b0.z), SIMD3(b1.x, b0.y, b1.z))
            seg(SIMD3(b1.x, b1.y, b0.z), SIMD3(b1.x, b1.y, b1.z))
            seg(SIMD3(b0.x, b1.y, b0.z), SIMD3(b0.x, b1.y, b1.z))

            if cfg.obstacleEnabled {
                let ox = cfg.obstacleX, oy = cfg.obstacleY, r = cfg.obstacleR
                let segs = 28
                for i in 0..<segs {
                    let a0 = Float(i) * (.pi * 2.0) / Float(segs)
                    let a1 = Float(i + 1) * (.pi * 2.0) / Float(segs)
                    let c0 = cos(a0) * r, s0 = sin(a0) * r
                    let c1 = cos(a1) * r, s1 = sin(a1) * r
                    seg(SIMD3(ox + c0, oy + s0, 0), SIMD3(ox + c1, oy + s1, 0))
                    seg(SIMD3(ox + c0, oy, s0), SIMD3(ox + c1, oy, s1))
                    seg(SIMD3(ox, oy + c0, s0), SIMD3(ox, oy + c1, s1))
                }
            }
        }

        let wireVerts = idx - gridVerts
        guard idx > 0 else { return }

        wireScratch.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                bufWire.contents().copyMemory(from: base, byteCount: idx * MemoryLayout<SIMD4<Float>>.stride)
            }
        }

        enc.setRenderPipelineState(pipeLine)
        enc.setDepthStencilState(depthDisabled)
        enc.setVertexBuffer(bufWire, offset: 0, index: 0)

        if gridVerts > 0 {
            var uGrid = makeRenderUniforms(lineColor: SIMD4<Float>(0.32, 0.45, 0.62, 0.28))
            enc.setVertexBytes(&uGrid, length: MemoryLayout<RenderUniformsGPU>.stride, index: 1)
            enc.setFragmentBytes(&uGrid, length: MemoryLayout<RenderUniformsGPU>.stride, index: 0)
            enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: gridVerts)
        }
        if wireVerts > 0 {
            var uWire = makeRenderUniforms(lineColor: SIMD4<Float>(0.68, 0.86, 1.00, 0.34))
            enc.setVertexBytes(&uWire, length: MemoryLayout<RenderUniformsGPU>.stride, index: 1)
            enc.setFragmentBytes(&uWire, length: MemoryLayout<RenderUniformsGPU>.stride, index: 0)
            enc.drawPrimitives(type: .line, vertexStart: gridVerts, vertexCount: wireVerts)
        }
    }

    private func encodeParticles(enc: MTLRenderCommandEncoder) {
        guard particleCount > 0,
              let bPos = bufPos,
              let bVel = bufVel,
              let pipe = active2D ? pipeParticle2D : pipeParticle3D,
              let dState = active2D ? depthDisabled : depthLessEqual else { return }

        var u = makeRenderUniforms()
        enc.setRenderPipelineState(pipe)
        enc.setDepthStencilState(dState)
        enc.setVertexBuffer(bPos, offset: 0, index: 0)
        enc.setVertexBuffer(bVel, offset: 0, index: 1)
        enc.setVertexBytes(&u, length: MemoryLayout<RenderUniformsGPU>.stride, index: 2)
        enc.setFragmentBytes(&u, length: MemoryLayout<RenderUniformsGPU>.stride, index: 0)

        enc.drawPrimitives(
            type: .triangleStrip,
            vertexStart: 0,
            vertexCount: 4,
            instanceCount: particleCount
        )
    }
}

struct MetalFluidViewport: UIViewRepresentable {
    @ObservedObject var cfg: SimulationConfig

    func makeCoordinator() -> MetalFluidEngine {
        MetalFluidEngine(cfg: cfg)
    }

    func makeUIView(context: Context) -> InteractiveMTKView {
        let mtk = InteractiveMTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        mtk.engine = context.coordinator
        mtk.delegate = context.coordinator
        mtk.colorPixelFormat = .bgra8Unorm
        mtk.depthStencilPixelFormat = .depth32Float
        mtk.preferredFramesPerSecond = 120
        mtk.isPaused = false
        mtk.enableSetNeedsDisplay = false
        return mtk
    }

    func updateUIView(_ uiView: InteractiveMTKView, context: Context) {}
}
