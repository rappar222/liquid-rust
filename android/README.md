# LiquidRust Studio — Android Mobile Edition (`.apk`)

Ultra-optimized real-time **Position-Based Fluids (PBF)** simulator built specifically for Android mobile GPUs (**Qualcomm Adreno**, **ARM Mali**, **Samsung Xclipse / RDNA**, **Imagination PowerVR**) with zero external library bloat.

## Built APK

- **Signed Release APK**: `dist/liquid-rust-android.apk`
- **Package ID**: `com.liquidrust.mobile`
- **Minimum Android Version**: Android 8.0 Oreo (`API 26`) — targets Android 14 (`API 34`)

### Install on a Connected Android Phone / Tablet

```bash
adb install -r dist/liquid-rust-android.apk
```

### Rebuild from Source

```bash
./build_apk.sh
```

You can also open this `android/` directory directly in **Android Studio**.

---

## Mobile Hardware Optimizations

1. **OpenGL ES 3.1 / 3.2 GPU Compute Pipeline (`GpuPbfSolver.java`)**:
   - **Workgroup Size `local_size_x = 64`**: Matches optimal 64-fiber Adreno waves and dual-warp Mali execution units without shared-memory register spilling.
   - **Strict `<= 4` SSBOs per Compute Shader**: Guaranteed hardware compliance across every OpenGL ES 3.1+ mobile GPU (`GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS >= 4`).
   - **L1-Cached Lock-Free Spatial Hash Grid**: Lock-free `atomicExchange` linked-list construction (`pbf_insert_grid.comp`) into a unified `grid_data` SSBO (`heads` + `next`), read via `readonly` non-atomic loads in neighbor passes so mobile texture/L1 caches coalesce reads.
   - **Fused Jacobi + Cohesion + Collider Pass (`pbf_solve_collide.comp`)**: Combines pressure `Delta p`, gated tensile `s_corr`, Akinci bell-spline surface cohesion, `0.65` Jacobi under-relaxation, and analytical chamber/sphere collision into a single compute pass.
   - **Anti-Crowding Floor Boundary Support (`pbf_density.comp`)**: Analytical mirror-floor Poly6 kernel density contribution prevents particles from crowding and popping along the bottom boundary.
   - **Zero-Copy SSBO-to-Vertex Rendering (`FluidRenderer.java`)**: Swaps `ssboPos` and `ssboPredA` handles in O(1) time and binds them directly to `GL_ARRAY_BUFFER` with `glVertexAttribDivisor(..., 1)` + `gl_VertexID` procedural billboards (`0` vertex SSBO blocks needed, `0` bytes copied between CPU and GPU).
   - **TBDR Early-Z Preservation (`particle.frag`)**: Compiles dedicated 2D and 3D shader variants so 2D Planar Mode avoids `gl_FragDepth` writes, preserving Tile-Based Deferred Rendering (TBDR) Early-Z / Forward Pixel Kill throughput.
2. **Multi-Core Zero-GC CPU Fallback (`CpuPbfSolver.java`)**:
   - Devices limited to OpenGL ES 3.0 automatically fall back to a work-partitioned multi-threaded flat-array PBF solver with zero per-frame heap allocations.
3. **90Hz / 120Hz High-Refresh Negotiation & Zero-GC Loop**:
   - Negotiates the highest supported display refresh rate on startup (`MainActivity.java`) and performs zero Java object allocations inside `onDrawFrame`.

---

## Mobile Controls & Features

- **2D Planar (Default) & 3D Volumetric Modes** across 5 quality tiers (`Draft` through `Extreme`) and all 6 presets (`Dam Break`, `Double Dam Break`, `Droplet Splash`, `Obstacle Drop`, `Vortex Tornado`, `Zero-Gravity Blobs`).
- **Hardware Tilt Gravity**: Toggle **Tilt Gravity** in the `Physics` tab (or top HUD) to pour, slosh, and shake the fluid in real time using your phone's accelerometer / gravity sensor.
- **Multi-Touch Gestures**:
  - **1-Finger Drag**: `Attract` (spring-damped fluid vortex), `Repel` (shockwave push), or `Orbit` (3D camera rotation), selectable from the bottom dock.
  - **2-Finger Pinch & Drag**: Smooth camera zoom and pan at any time.
- **Live Optical Material Studio**: 8 physical presets (`Clear Water`, `Deep Ocean`, `Tropical Lagoon`, `Molten Amber`, `Crimson Wine`, `Emerald Gel`, `Creamy Milk`, `Liquid Chrome`), procedural `Ripples` / `Caustics` / `Foam` / `Grid` textures, Fresnel IOR, GGX Roughness, Specular, and Subsurface Refraction controls.
