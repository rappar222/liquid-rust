// LiquidRust Studio Web Application Controller
// Dual engine: WebGPU Compute Shaders (primary) with WebGL 2.0 fallback,
// Live Optical Material / Procedural Texture Shading, Natural Fluid Physics, and OS-Aware Native Downloads.

import { Vec3, Ray } from './math.js?v=20261113b';
import { Camera } from './camera.js?v=20261113b';
import { PbfSolver, ScenePreset } from './pbf-solver.js?v=20261113b';
import { GlRenderer } from './gl-renderer.js?v=20261113b';
import { WebGpuSolver } from './webgpu-solver.js?v=20261113b';
import { WebGpuRenderer } from './webgpu-renderer.js?v=20261113b';

const RELEASE_LATEST = 'https://github.com/rappar222/liquid-rust/releases/latest';
const RELEASE_DL = `${RELEASE_LATEST}/download`;
const RELEASE_V033_DL = 'https://github.com/rappar222/liquid-rust/releases/download/v0.3.3';
const ANDROID_APK_URL = `${RELEASE_DL}/liquid-rust-android.apk`;
const IOS_IPA_URL = `${RELEASE_DL}/liquid-rust-ios.ipa`;

const MATERIAL_PRESETS = {
    water:   { shallow: '#38bdf8', deep: '#0c3b8a', ior: 1.333, roughness: 0.045, specular: 1.35, refraction: 0.42, metallic: 0.00, foam: 0.35 },
    ocean:   { shallow: '#2199d6', deep: '#031733', ior: 1.340, roughness: 0.065, specular: 1.30, refraction: 0.38, metallic: 0.00, foam: 0.55 },
    lagoon:  { shallow: '#3de8db', deep: '#055261', ior: 1.335, roughness: 0.040, specular: 1.40, refraction: 0.48, metallic: 0.00, foam: 0.25 },
    amber:   { shallow: '#fcb82e', deep: '#5c1403', ior: 1.540, roughness: 0.055, specular: 1.45, refraction: 0.52, metallic: 0.00, foam: 0.12 },
    wine:    { shallow: '#c71f47', deep: '#24030d', ior: 1.360, roughness: 0.050, specular: 1.30, refraction: 0.32, metallic: 0.00, foam: 0.10 },
    emerald: { shallow: '#33eb85', deep: '#03381f', ior: 1.520, roughness: 0.042, specular: 1.50, refraction: 0.48, metallic: 0.00, foam: 0.15 },
    milk:    { shallow: '#fafcff', deep: '#d9e3f0', ior: 1.350, roughness: 0.180, specular: 0.85, refraction: 0.08, metallic: 0.00, foam: 0.05 },
    chrome:  { shallow: '#e6ecf5', deep: '#6a788e', ior: 2.400, roughness: 0.025, specular: 2.30, refraction: 0.00, metallic: 0.94, foam: 0.00 },
};

const TEX_NAME_TO_ID = {
    none: 0,
    ripples: 1,
    caustics: 2,
    foam: 3,
    grid: 4,
};

function hexToRgb(hex) {
    const clean = (hex || '#38bdf8').replace('#', '');
    const full = clean.length === 3 ? clean.split('').map((c) => c + c).join('') : clean;
    const n = parseInt(full, 16) || 0;
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

class LiquidApp {
    constructor() {
        this.canvas = document.getElementById('glcanvas');
        this.camera = new Camera();

        // Viewport & Physics state (defaults: 2D Planar Mode + Draft Lattice)
        this.isPlaying = true;
        this.simSpeed = 1.0;
        this.currentPreset = ScenePreset.DAM_BREAK;
        this.backend = 'webgl';
        this.gpuHint = '';

        // Live Optical & Procedural Shading Uniforms
        this.renderOpts = {
            colorMode: 'material',
            colorMapType: 'ocean',
            litSpheres: true,
            shallowColor: hexToRgb('#38bdf8'),
            deepColor: hexToRgb('#0c3b8a'),
            texMode: 1,
            texStrength: 0.28,
            ior: 1.333,
            roughness: 0.045,
            specular: 1.35,
            refraction: 0.42,
            metallic: 0.0,
            foamAmount: 0.35,
            animTime: 0.0,
        };

        // Pointer interaction state
        this.isPointerDown = false;
        this.pointerButton = 0;
        this.lastPointerX = 0;
        this.lastPointerY = 0;
        this.isShiftHeld = false;

        // Rolling telemetry window
        this.fpsFrameCount = 0;
        this.fpsLastTime = performance.now();
        this.currentFps = 60;
        this.simStepMs = 0;
        this.simStepAccum = 0;
        this.simStepCount = 0;

        this.initUI();
        this.detectOsDownload();
        this.initEvents();
        this.resize();
    }

    async init() {
        let gpuAvailable = false;

        // 1. Try WebGPU Compute Pipeline first
        if ('gpu' in navigator) {
            try {
                let adapter = await navigator.gpu.requestAdapter({
                    powerPreference: 'high-performance'
                });
                if (!adapter) {
                    adapter = await navigator.gpu.requestAdapter();
                }

                if (adapter) {
                    const device = await adapter.requestDevice();
                    const gpuContext = this.canvas.getContext('webgpu');
                    if (gpuContext) {
                        const format = navigator.gpu.getPreferredCanvasFormat();
                        gpuContext.configure({
                            device,
                            format,
                            alphaMode: 'premultiplied',
                        });

                        this.solver = new WebGpuSolver(device);
                        this.renderer = new WebGpuRenderer(device, gpuContext, format);
                        this.backend = 'webgpu';
                        gpuAvailable = true;

                        const info = adapter.info || {};
                        const gpuLabel = info.device || info.description || 'GPU Compute';
                        this.gpuHint = `${info.vendor || ''} ${info.architecture || ''} ${info.description || ''} ${info.device || ''}`;
                        this.updateBackendBadge(`⚡ WebGPU · ${gpuLabel}`, 'webgpu');
                    }
                }
            } catch (err) {
                console.warn('WebGPU init failed, falling back to WebGL 2.0:', err);
            }
        }

        // 2. Fallback to WebGL 2.0 (Optimized CPU PBF Solver + Glossy GLSL 3.00 ES Renderer)
        if (!gpuAvailable) {
            let reason = 'CPU Fallback';
            if (!window.isSecureContext) {
                reason = 'HTTPS/localhost needed for WebGPU';
            } else if (!('gpu' in navigator)) {
                reason = 'WebGPU disabled';
            }

            this.gl = this.canvas.getContext('webgl2', {
                antialias: true,
                depth: true,
                premultipliedAlpha: false
            });

            if (!this.gl) {
                document.body.innerHTML = `
                    <div style="color:#eef3fb;padding:48px;font-family:sans-serif;text-align:center;">
                        <h2>WebGL 2.0 / WebGPU Not Supported</h2>
                        <p style="margin-top:10px;color:#98a7c2;">Please enable hardware acceleration in your browser settings.</p>
                    </div>
                `;
                return;
            }

            try {
                const dbg = this.gl.getExtension('WEBGL_debug_renderer_info');
                if (dbg) {
                    this.gpuHint = this.gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || '';
                }
            } catch { /* ignore debug renderer failure */ }

            this.solver = new PbfSolver();
            this.renderer = new GlRenderer(this.gl);
            this.backend = 'webgl';
            this.updateBackendBadge(`WebGL 2.0 · ${reason}`, 'webgl');
        }

        // Refine OS / Arch download detection using GPU & high-entropy UA hints
        this.detectOsDownload(this.gpuHint);

        // Lock defaults to 2D Planar Mode & Draft Quality
        const is2D = document.getElementById('toggle-2d').checked;
        const quality = document.getElementById('select-quality').value;
        this.solver.set2D(is2D);
        this.solver.setQuality(quality);
        this.camera.reset(is2D);
        this.solver.loadPreset(this.currentPreset);
        this.syncObstacleFromSolver();
        this.syncKernelReadout();
        this.updateStats(60);

        requestAnimationFrame((t) => this.renderLoop(t));
    }

    updateBackendBadge(text, cls) {
        const badge = document.getElementById('badge-backend');
        if (badge) {
            badge.textContent = text;
            badge.className = 'badge-backend ' + cls;
            badge.title = text;
        }
    }

    syncKernelReadout() {
        const el = document.getElementById('val-kernel-h');
        if (el && this.solver) {
            el.textContent = `${this.solver.h.toFixed(3)} m`;
        }
    }

    syncObstacleFromSolver() {
        if (!this.solver) return;
        document.getElementById('check-obstacle').checked = !!this.solver.obstacleEnabled;
        document.getElementById('slider-obs-x').value = this.solver.obstaclePos.x;
        document.getElementById('val-obs-x').textContent = this.solver.obstaclePos.x.toFixed(2) + ' m';
        document.getElementById('slider-obs-y').value = this.solver.obstaclePos.y;
        document.getElementById('val-obs-y').textContent = this.solver.obstaclePos.y.toFixed(2) + ' m';
        document.getElementById('slider-obs-r').value = this.solver.obstacleRadius;
        document.getElementById('val-obs-r').textContent = this.solver.obstacleRadius.toFixed(2) + ' m';

        const gMag = Math.abs(this.solver.gravity.y);
        document.getElementById('slider-gravity').value = gMag;
        document.getElementById('val-gravity').textContent = gMag.toFixed(1) + ' m/s²';
    }

    updateStats(fps = null) {
        if (fps !== null) {
            document.getElementById('stat-fps').textContent = fps;
        }
        if (this.solver) {
            document.getElementById('stat-particles').textContent = this.solver.particleCount.toLocaleString();
        }
        document.getElementById('stat-step-time').textContent = this.simStepMs.toFixed(1) + ' ms';
    }

    resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
        const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));

        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.canvas.width = w;
            this.canvas.height = h;
        }
    }

    initUI() {
        // Studio Inspector Tabs
        const tabs = document.querySelectorAll('.studio-tab');
        const panes = document.querySelectorAll('.tab-pane');
        tabs.forEach((btn) => {
            btn.addEventListener('click', () => {
                const targetId = btn.dataset.tab;
                tabs.forEach((t) => {
                    const active = t === btn;
                    t.classList.toggle('active', active);
                    t.setAttribute('aria-selected', active ? 'true' : 'false');
                });
                panes.forEach((p) => {
                    p.classList.toggle('active', p.id === targetId);
                });
            });
        });

        // Transport Controls
        const btnPlay = document.getElementById('btn-play');
        btnPlay.addEventListener('click', () => {
            this.isPlaying = !this.isPlaying;
            btnPlay.innerHTML = this.isPlaying ? '<span>⏸</span> Pause' : '<span>▶</span> Play';
        });

        document.getElementById('btn-step').addEventListener('click', () => {
            if (!this.solver) return;
            this.isPlaying = false;
            btnPlay.innerHTML = '<span>▶</span> Play';
            this.solver.step(1.0 / 60.0, this.simSpeed);
            this.renderOpts.animTime += (1.0 / 60.0) * this.simSpeed;
            this.updateStats();
        });

        document.getElementById('btn-reset').addEventListener('click', () => {
            if (!this.solver) return;
            this.solver.loadPreset(this.currentPreset);
            this.syncObstacleFromSolver();
            this.updateStats();
        });

        document.getElementById('btn-frame').addEventListener('click', () => {
            if (!this.solver) return;
            this.camera.reset(this.solver.is2D);
        });

        // Scene Preset
        const presetSelect = document.getElementById('select-preset');
        presetSelect.addEventListener('change', (e) => {
            this.currentPreset = e.target.value;
            if (this.solver) {
                this.solver.loadPreset(this.currentPreset);
                this.syncObstacleFromSolver();
                this.updateStats();
            }
        });

        // 2D / 3D Mode Switch
        const mode2D = document.getElementById('toggle-2d');
        const qualitySelect = document.getElementById('select-quality');
        mode2D.addEventListener('change', (e) => {
            const is2D = e.target.checked;
            if (this.solver) {
                this.solver.set2D(is2D);
                this.solver.setQuality(qualitySelect.value);
                this.camera.reset(is2D);
                this.solver.loadPreset(this.currentPreset);
                this.syncObstacleFromSolver();
                this.syncKernelReadout();
                this.updateStats();
            }
            document.getElementById('mode-label').textContent = is2D ? '2D Planar Mode' : '3D Fluid Viewport';
        });

        // Time Scale & ASMR Chips
        const speedSlider = document.getElementById('slider-speed');
        const speedValue = document.getElementById('val-speed');
        const chipAsmr = document.getElementById('btn-speed-asmr');
        const chipSmooth = document.getElementById('btn-speed-smooth');
        const chipReal = document.getElementById('btn-speed-real');

        const setSpeed = (s) => {
            this.simSpeed = s;
            speedSlider.value = s;
            speedValue.textContent = s.toFixed(2) + 'x';
            chipAsmr.classList.toggle('active', Math.abs(s - 0.25) < 0.01);
            chipSmooth.classList.toggle('active', Math.abs(s - 0.50) < 0.01);
            chipReal.classList.toggle('active', Math.abs(s - 1.00) < 0.01);
        };

        speedSlider.addEventListener('input', (e) => setSpeed(parseFloat(e.target.value)));
        chipAsmr.addEventListener('click', () => setSpeed(0.25));
        chipSmooth.addEventListener('click', () => setSpeed(0.50));
        chipReal.addEventListener('click', () => setSpeed(1.00));

        // Lattice Quality
        qualitySelect.addEventListener('change', (e) => {
            if (!this.solver) return;
            this.solver.setQuality(e.target.value);
            this.solver.loadPreset(this.currentPreset);
            this.syncObstacleFromSolver();
            this.syncKernelReadout();
            this.updateStats();
        });

        // Natural Fluid Dynamics Sliders
        const vortSlider = document.getElementById('slider-vorticity');
        const vortVal = document.getElementById('val-vorticity');
        vortSlider.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            if (this.solver) this.solver.vorticity = v;
            vortVal.textContent = v.toFixed(3);
        });

        const cohSlider = document.getElementById('slider-cohesion');
        const cohVal = document.getElementById('val-cohesion');
        cohSlider.addEventListener('input', (e) => {
            const c = parseFloat(e.target.value);
            if (this.solver) this.solver.surfaceTension = c;
            cohVal.textContent = c.toFixed(3);
        });

        const viscSlider = document.getElementById('slider-visc');
        const viscVal = document.getElementById('val-visc');
        viscSlider.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            if (this.solver) this.solver.viscosity = v;
            viscVal.textContent = v.toFixed(3);
        });

        const gravitySlider = document.getElementById('slider-gravity');
        const gravityVal = document.getElementById('val-gravity');
        gravitySlider.addEventListener('input', (e) => {
            const g = parseFloat(e.target.value);
            if (this.solver) this.solver.gravity.y = -g;
            gravityVal.textContent = g.toFixed(1) + ' m/s²';
        });

        // Colliders & Obstacles
        document.getElementById('check-obstacle').addEventListener('change', (e) => {
            if (this.solver) this.solver.obstacleEnabled = e.target.checked;
        });

        document.getElementById('slider-obs-x').addEventListener('input', (e) => {
            const x = parseFloat(e.target.value);
            if (this.solver) this.solver.obstaclePos.x = x;
            document.getElementById('val-obs-x').textContent = x.toFixed(2) + ' m';
        });
        document.getElementById('slider-obs-y').addEventListener('input', (e) => {
            const y = parseFloat(e.target.value);
            if (this.solver) this.solver.obstaclePos.y = y;
            document.getElementById('val-obs-y').textContent = y.toFixed(2) + ' m';
        });
        document.getElementById('slider-obs-r').addEventListener('input', (e) => {
            const r = parseFloat(e.target.value);
            if (this.solver) this.solver.obstacleRadius = r;
            document.getElementById('val-obs-r').textContent = r.toFixed(2) + ' m';
        });

        // Live Optical Material & Surface Shading
        const colorModeSelect = document.getElementById('select-color-mode');
        const colorMapGroup = document.getElementById('colormap-group');
        const syncColorModeUI = (mode) => {
            this.renderOpts.colorMode = mode;
            colorModeSelect.value = mode;
            colorMapGroup.classList.toggle('hidden', mode === 'material');
        };

        colorModeSelect.addEventListener('change', (e) => {
            syncColorModeUI(e.target.value);
        });

        document.getElementById('select-color-map').addEventListener('change', (e) => {
            this.renderOpts.colorMapType = e.target.value;
        });

        document.getElementById('check-lit-spheres').addEventListener('change', (e) => {
            this.renderOpts.litSpheres = e.target.checked;
        });

        const matPreset = document.getElementById('mat-preset');
        const matShallow = document.getElementById('mat-color-shallow');
        const matDeep = document.getElementById('mat-color-deep');
        const valShallow = document.getElementById('val-mat-shallow');
        const valDeep = document.getElementById('val-mat-deep');
        const matTex = document.getElementById('mat-texture');
        const matTexStr = document.getElementById('mat-tex-strength');
        const matFoam = document.getElementById('mat-foam');
        const matIor = document.getElementById('mat-ior');
        const matRough = document.getElementById('mat-roughness');
        const matSpec = document.getElementById('mat-specular');
        const matRefr = document.getElementById('mat-refraction');
        const matMetal = document.getElementById('mat-metallic');

        const ensureMaterialMode = () => {
            if (this.renderOpts.colorMode !== 'material') {
                syncColorModeUI('material');
            }
        };

        matPreset.addEventListener('change', (e) => {
            const p = MATERIAL_PRESETS[e.target.value];
            ensureMaterialMode();
            if (!p) return;

            matShallow.value = p.shallow;
            matDeep.value = p.deep;
            valShallow.textContent = p.shallow.toUpperCase();
            valDeep.textContent = p.deep.toUpperCase();
            matIor.value = p.ior;
            matRough.value = p.roughness;
            matSpec.value = p.specular;
            matRefr.value = p.refraction;
            matMetal.value = p.metallic;
            matFoam.value = p.foam;

            document.getElementById('val-mat-ior').textContent = p.ior.toFixed(3);
            document.getElementById('val-mat-rough').textContent = p.roughness.toFixed(3);
            document.getElementById('val-mat-spec').textContent = p.specular.toFixed(2);
            document.getElementById('val-mat-refr').textContent = p.refraction.toFixed(2);
            document.getElementById('val-mat-metallic').textContent = p.metallic.toFixed(2);
            document.getElementById('val-mat-foam').textContent = p.foam.toFixed(2);

            this.renderOpts.shallowColor = hexToRgb(p.shallow);
            this.renderOpts.deepColor = hexToRgb(p.deep);
            this.renderOpts.ior = p.ior;
            this.renderOpts.roughness = p.roughness;
            this.renderOpts.specular = p.specular;
            this.renderOpts.refraction = p.refraction;
            this.renderOpts.metallic = p.metallic;
            this.renderOpts.foamAmount = p.foam;
        });

        const onCustomColor = () => {
            ensureMaterialMode();
            matPreset.value = 'custom';
            valShallow.textContent = matShallow.value.toUpperCase();
            valDeep.textContent = matDeep.value.toUpperCase();
            this.renderOpts.shallowColor = hexToRgb(matShallow.value);
            this.renderOpts.deepColor = hexToRgb(matDeep.value);
        };
        matShallow.addEventListener('input', onCustomColor);
        matDeep.addEventListener('input', onCustomColor);

        matTex.addEventListener('change', (e) => {
            this.renderOpts.texMode = TEX_NAME_TO_ID[e.target.value] ?? 0;
        });

        matTexStr.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            this.renderOpts.texStrength = v;
            document.getElementById('val-mat-tex').textContent = v.toFixed(2);
        });

        matFoam.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            this.renderOpts.foamAmount = v;
            document.getElementById('val-mat-foam').textContent = v.toFixed(2);
        });

        matIor.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            this.renderOpts.ior = v;
            document.getElementById('val-mat-ior').textContent = v.toFixed(3);
        });

        matRough.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            this.renderOpts.roughness = v;
            document.getElementById('val-mat-rough').textContent = v.toFixed(3);
        });

        matSpec.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            this.renderOpts.specular = v;
            document.getElementById('val-mat-spec').textContent = v.toFixed(2);
        });

        matRefr.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            this.renderOpts.refraction = v;
            document.getElementById('val-mat-refr').textContent = v.toFixed(2);
        });

        matMetal.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            this.renderOpts.metallic = v;
            document.getElementById('val-mat-metallic').textContent = v.toFixed(2);
        });

        // Collapsible Inspector Sections
        document.querySelectorAll('.panel-header').forEach((header) => {
            header.addEventListener('click', () => {
                header.parentElement.classList.toggle('collapsed');
            });
        });

        // Keyboard Shortcuts
        window.addEventListener('keydown', (e) => {
            if (e.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
            if (e.code === 'Space') {
                e.preventDefault();
                btnPlay.click();
            } else if (e.code === 'KeyR') {
                if (this.solver) {
                    this.solver.loadPreset(this.currentPreset);
                    this.syncObstacleFromSolver();
                    this.updateStats();
                }
            } else if (e.code === 'KeyC') {
                if (this.solver) {
                    this.camera.reset(this.solver.is2D);
                }
            }
        });
    }

    async detectOsDownload(gpuHint = '') {
        const ua = navigator.userAgent || '';
        const plat = navigator.platform || '';
        let uaArch = '';
        let uaPlatform = '';

        if (navigator.userAgentData) {
            uaPlatform = navigator.userAgentData.platform || '';
            if (typeof navigator.userAgentData.getHighEntropyValues === 'function') {
                try {
                    const high = await navigator.userAgentData.getHighEntropyValues(['architecture', 'platform']);
                    uaArch = high.architecture || '';
                    uaPlatform = high.platform || uaPlatform;
                } catch { /* ignore high-entropy rejection */ }
            }
        }

        const p = (uaPlatform || plat || ua).toLowerCase();
        const combined = `${ua} ${plat} ${uaArch} ${gpuHint}`.toLowerCase();
        const isArm = /arm64|aarch64|apple m\d/.test(combined);
        const isAndroid = /android/.test(ua.toLowerCase());
        const isIos = /iphone|ipad|ipod/.test(ua.toLowerCase()) || (p.includes('mac') && (navigator.maxTouchPoints || 0) > 1);

        let osLabel = 'Desktop';
        let primaryLabel = '⬇ Download LiquidRust';
        let primaryHref = RELEASE_LATEST;
        let headerLabel = '⬇ Get App';
        let alts = [];

        if (isAndroid) {
            osLabel = 'Android 8.0+';
            primaryLabel = '⬇ Download Android (.apk)';
            primaryHref = ANDROID_APK_URL;
            headerLabel = '⬇ Android APK';
            alts = [
                { label: 'Windows .zip', href: `${RELEASE_DL}/liquid-rust-windows-x86_64.zip` },
                { label: 'Linux AppImage', href: `${RELEASE_DL}/liquid-rust-x86_64.AppImage` },
                { label: 'macOS ARM64', href: `${RELEASE_DL}/liquid-rust-macos-arm64.tar.gz` },
            ];
        } else if (isIos) {
            osLabel = 'iOS / iPadOS 16.4+';
            primaryLabel = '⬇ Download iOS (.ipa)';
            primaryHref = IOS_IPA_URL;
            headerLabel = '⬇ iOS IPA';
            alts = [
                { label: 'Android .apk', href: ANDROID_APK_URL },
                { label: 'macOS ARM64', href: `${RELEASE_DL}/liquid-rust-macos-arm64.tar.gz` },
                { label: 'Windows .zip', href: `${RELEASE_DL}/liquid-rust-windows-x86_64.zip` },
            ];
        } else if (p.includes('win')) {
            osLabel = 'Windows x86_64';
            primaryLabel = '⬇ Download Windows (.zip)';
            primaryHref = `${RELEASE_DL}/liquid-rust-windows-x86_64.zip`;
            headerLabel = '⬇ Windows App';
            alts = [
                { label: 'Android .apk', href: ANDROID_APK_URL },
                { label: 'All Releases', href: RELEASE_LATEST },
            ];
        } else if (p.includes('mac') || p.includes('darwin')) {
            if (isArm) {
                osLabel = 'macOS Apple Silicon';
                primaryLabel = '⬇ Download macOS (ARM64)';
                primaryHref = `${RELEASE_DL}/liquid-rust-macos-arm64.tar.gz`;
                headerLabel = '⬇ macOS App';
                alts = [
                    { label: 'Intel Mac (.tar.gz)', href: `${RELEASE_DL}/liquid-rust-macos-x86_64.tar.gz` },
                    { label: 'All Releases', href: RELEASE_LATEST },
                ];
            } else {
                osLabel = 'macOS Intel x86_64';
                primaryLabel = '⬇ Download macOS (Intel)';
                primaryHref = `${RELEASE_DL}/liquid-rust-macos-x86_64.tar.gz`;
                headerLabel = '⬇ macOS App';
                alts = [
                    { label: 'Apple Silicon (.tar.gz)', href: `${RELEASE_DL}/liquid-rust-macos-arm64.tar.gz` },
                    { label: 'All Releases', href: RELEASE_LATEST },
                ];
            }
        } else if (p.includes('linux') || p.includes('x11')) {
            if (isArm) {
                osLabel = 'Linux ARM64';
                primaryLabel = '⬇ Download Linux ARM64 (.tar.gz)';
                primaryHref = `${RELEASE_DL}/liquid-rust-linux-aarch64.tar.gz`;
                headerLabel = '⬇ Linux ARM64';
                alts = [
                    { label: '.rpm (aarch64)', href: `${RELEASE_V033_DL}/liquid-rust-0.3.3-1.aarch64.rpm` },
                    { label: 'x86_64 AppImage', href: `${RELEASE_DL}/liquid-rust-x86_64.AppImage` },
                    { label: 'All Releases', href: RELEASE_LATEST },
                ];
            } else {
                const uaLow = ua.toLowerCase();
                if (/ubuntu|debian|mint|pop!_os/.test(uaLow)) {
                    osLabel = 'Ubuntu / Debian x64';
                    primaryLabel = '⬇ Download Ubuntu (.deb)';
                    primaryHref = `${RELEASE_V033_DL}/liquid-rust_0.3.3_amd64.deb`;
                    alts = [
                        { label: 'AppImage', href: `${RELEASE_DL}/liquid-rust-x86_64.AppImage` },
                        { label: 'Android .apk', href: ANDROID_APK_URL },
                        { label: 'All Releases', href: RELEASE_LATEST },
                    ];
                } else if (/fedora|rhel|centos|rocky|alma/.test(uaLow)) {
                    osLabel = 'Fedora / RHEL x64';
                    primaryLabel = '⬇ Download Fedora (.rpm)';
                    primaryHref = `${RELEASE_V033_DL}/liquid-rust-0.3.3-1.x86_64.rpm`;
                    alts = [
                        { label: 'AppImage', href: `${RELEASE_DL}/liquid-rust-x86_64.AppImage` },
                        { label: 'Android .apk', href: ANDROID_APK_URL },
                        { label: 'All Releases', href: RELEASE_LATEST },
                    ];
                } else {
                    osLabel = 'Linux x86_64';
                    primaryLabel = '⬇ Download Linux (AppImage)';
                    primaryHref = `${RELEASE_DL}/liquid-rust-x86_64.AppImage`;
                    alts = [
                        { label: '.deb', href: `${RELEASE_V033_DL}/liquid-rust_0.3.3_amd64.deb` },
                        { label: '.rpm', href: `${RELEASE_V033_DL}/liquid-rust-0.3.3-1.x86_64.rpm` },
                        { label: 'Android .apk', href: ANDROID_APK_URL },
                        { label: 'All Releases', href: RELEASE_LATEST },
                    ];
                }
                headerLabel = '⬇ Linux App';
            }
        } else {
            alts = [
                { label: 'Windows .zip', href: `${RELEASE_DL}/liquid-rust-windows-x86_64.zip` },
                { label: 'Linux AppImage', href: `${RELEASE_DL}/liquid-rust-x86_64.AppImage` },
                { label: 'macOS ARM64', href: `${RELEASE_DL}/liquid-rust-macos-arm64.tar.gz` },
            ];
        }

        const pill = document.getElementById('download-os-label');
        const btn = document.getElementById('btn-download-app');
        const hdrBtn = document.getElementById('header-download-btn');
        const altWrap = document.getElementById('download-alt-links');

        if (pill) pill.textContent = osLabel;
        if (btn) {
            btn.textContent = primaryLabel;
            btn.href = primaryHref;
        }
        if (hdrBtn) {
            hdrBtn.textContent = headerLabel;
            hdrBtn.href = primaryHref;
            hdrBtn.title = `${primaryLabel.replace('⬇ ', '')} (${osLabel})`;
        }
        if (altWrap) {
            altWrap.innerHTML = '';
            for (const item of alts) {
                const a = document.createElement('a');
                a.className = 'dl-alt-link';
                a.href = item.href;
                a.target = '_blank';
                a.rel = 'noopener';
                a.textContent = item.label;
                altWrap.appendChild(a);
            }
        }
    }

    initEvents() {
        window.addEventListener('resize', () => this.resize());

        window.addEventListener('keydown', (e) => {
            if (e.key === 'Shift') {
                this.isShiftHeld = true;
                if (this.solver && this.isPointerDown && this.pointerButton === 0) {
                    this.solver.mouseForceMode = -1;
                }
            }
        });

        window.addEventListener('keyup', (e) => {
            if (e.key === 'Shift') {
                this.isShiftHeld = false;
                if (this.solver && this.isPointerDown && this.pointerButton === 0) {
                    this.solver.mouseForceMode = 1;
                }
            }
        });

        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

        this.canvas.addEventListener('mousedown', (e) => {
            if (e.button === 1) {
                e.preventDefault(); // Prevent middle-click browser auto-scroll
            }
            this.isPointerDown = true;
            this.pointerButton = e.button;
            this.isShiftHeld = !!e.shiftKey;
            this.lastPointerX = e.clientX;
            this.lastPointerY = e.clientY;

            if (this.solver) {
                // Left Drag = Attract (1), Shift + Left Drag = Repel (-1)
                if (e.button === 0) {
                    this.solver.mouseForceMode = this.isShiftHeld ? -1 : 1;
                    this.updateMouseRay(e.clientX, e.clientY);
                } else {
                    this.solver.mouseForceMode = 0;
                }
            }
        });

        window.addEventListener('mouseup', () => {
            this.isPointerDown = false;
            if (this.solver) this.solver.mouseForceMode = 0;
        });

        window.addEventListener('mousemove', (e) => {
            this.isShiftHeld = !!e.shiftKey;

            if (!this.isPointerDown || !this.solver) return;

            const dx = e.clientX - this.lastPointerX;
            const dy = e.clientY - this.lastPointerY;
            this.lastPointerX = e.clientX;
            this.lastPointerY = e.clientY;

            if (this.pointerButton === 0) {
                this.solver.mouseForceMode = this.isShiftHeld ? -1 : 1;
                this.updateMouseRay(e.clientX, e.clientY);
            } else if (this.pointerButton === 1) {
                // Middle-Click Drag: always pan camera
                this.camera.pan(dx, dy);
            } else if (this.pointerButton === 2 && !this.solver.is2D) {
                // Right-Click Drag: orbit 3D camera
                this.camera.rotate(dx * 0.006, -dy * 0.006);
            }
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.camera.zoom(e.deltaY * 0.0015);
        }, { passive: false });

        let initialTouchDist = 0;
        this.canvas.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                this.isPointerDown = true;
                this.pointerButton = 0;
                this.lastPointerX = e.touches[0].clientX;
                this.lastPointerY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                initialTouchDist = Math.hypot(
                    e.touches[0].clientX - e.touches[1].clientX,
                    e.touches[0].clientY - e.touches[1].clientY
                );
            }
        });

        this.canvas.addEventListener('touchmove', (e) => {
            if (!this.solver) return;
            if (e.touches.length === 1 && this.isPointerDown) {
                const dx = e.touches[0].clientX - this.lastPointerX;
                const dy = e.touches[0].clientY - this.lastPointerY;
                this.lastPointerX = e.touches[0].clientX;
                this.lastPointerY = e.touches[0].clientY;
                if (this.solver.is2D) {
                    this.camera.pan(dx, dy);
                } else {
                    this.camera.rotate(dx * 0.006, -dy * 0.006);
                }
            } else if (e.touches.length === 2) {
                const dist = Math.hypot(
                    e.touches[0].clientX - e.touches[1].clientX,
                    e.touches[0].clientY - e.touches[1].clientY
                );
                const delta = (dist - initialTouchDist) * 0.02;
                this.camera.zoom(-delta);
                initialTouchDist = dist;
            }
        });

        this.canvas.addEventListener('touchend', () => {
            this.isPointerDown = false;
        });
    }

    updateMouseRay(clientX, clientY) {
        if (!this.solver) return;
        const rect = this.canvas.getBoundingClientRect();
        const x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
        const y = -(((clientY - rect.top) / Math.max(1, rect.height)) * 2 - 1);

        const aspect = Math.max(0.1, rect.width / Math.max(1, rect.height));
        const viewMat = this.camera.viewMatrix();
        const projMat = this.camera.projectionMatrix(aspect);
        const viewProjInv = projMat.clone().multiply(viewMat).invert();

        const pNear = this.transformVec4(viewProjInv, [x, y, -1.0, 1.0]);
        const pFar = this.transformVec4(viewProjInv, [x, y, 1.0, 1.0]);

        const origin = new Vec3(pNear.x, pNear.y, pNear.z);
        const dir = new Vec3(pFar.x - pNear.x, pFar.y - pNear.y, pFar.z - pNear.z).normalize();
        this.solver.mouseRay = new Ray(origin, dir);
    }

    transformVec4(mat, v) {
        const m = mat.elements;
        const x = m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3];
        const y = m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3];
        const z = m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3];
        const w = m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3];
        const invW = 1.0 / (w || 1.0);
        return { x: x * invW, y: y * invW, z: z * invW };
    }

    renderLoop() {
        if (this.isPlaying) {
            const t0 = performance.now();
            this.solver.step(1.0 / 60.0, this.simSpeed);
            this.renderOpts.animTime += (1.0 / 60.0) * this.simSpeed;
            const stepMs = performance.now() - t0;
            this.simStepAccum += stepMs;
            this.simStepCount++;
        }

        this.renderer.render(
            this.solver,
            this.camera,
            this.canvas.width,
            this.canvas.height,
            this.renderOpts
        );

        this.fpsFrameCount++;
        const now = performance.now();
        const elapsed = now - this.fpsLastTime;

        if (elapsed >= 500) {
            this.currentFps = Math.round((this.fpsFrameCount * 1000) / elapsed);
            this.simStepMs = this.simStepCount > 0 ? (this.simStepAccum / this.simStepCount) : 0;
            this.updateStats(this.currentFps);
            this.fpsFrameCount = 0;
            this.fpsLastTime = now;
            this.simStepAccum = 0;
            this.simStepCount = 0;
        }

        requestAnimationFrame((t) => this.renderLoop(t));
    }
}

async function startApp() {
    try {
        window.app = new LiquidApp();
        await window.app.init();
    } catch (err) {
        console.error('Failed to start LiquidRust:', err);
    }
}

if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', startApp);
} else {
    startApp();
}
